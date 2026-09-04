use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::{
    env, fs,
    io::{self, BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
    sync::Arc,
};

const MOVIE_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "avi", "mov", "webm", "m4v", "mpg", "mpeg", "wmv", "flv",
];

fn print_usage(program: &str) {
    eprintln!(
        "Usage: {program} [--check] [movie-or-directory] [output-directory]\n\n\
         With no path, scans ~/Downloads/Movies. A directory scan includes movies directly\n\
         inside it and movies inside its immediate subdirectories. --check lists the movies\n\
         and asks for confirmation before starting. Screenshots are saved every 0.5 seconds."
    );
}

fn home_dir() -> Result<PathBuf, String> {
    env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| "HOME is not set; provide paths explicitly".to_owned())
}

fn default_movies_dir() -> Result<PathBuf, String> {
    Ok(home_dir()?.join("Downloads/Movies"))
}

fn default_output_dir() -> Result<PathBuf, String> {
    Ok(home_dir()?.join("curtain/screenshots"))
}

fn confirm(movies: &[PathBuf]) -> Result<bool, String> {
    println!("Found {} movie(s):", movies.len());
    for movie in movies {
        println!("  {}", movie.display());
    }

    print!("\nCreate screenshots for these movies? [y/N] ");
    io::stdout()
        .flush()
        .map_err(|error| format!("could not write prompt: {error}"))?;
    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .map_err(|error| format!("could not read answer: {error}"))?;
    Ok(matches!(
        answer.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

fn is_movie(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            MOVIE_EXTENSIONS
                .iter()
                .any(|candidate| extension.eq_ignore_ascii_case(candidate))
        })
}

fn find_movies(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = fs::read_dir(directory)
        .map_err(|error| format!("could not read {}: {error}", directory.display()))?;
    let mut movies = Vec::new();

    for entry in entries {
        let path = entry
            .map_err(|error| format!("could not read a directory entry: {error}"))?
            .path();

        if path.is_file() && is_movie(&path) {
            movies.push(path);
        } else if path.is_dir() {
            let children = fs::read_dir(&path)
                .map_err(|error| format!("could not read {}: {error}", path.display()))?;
            for child in children {
                let child = child
                    .map_err(|error| format!("could not read a directory entry: {error}"))?
                    .path();
                if child.is_file() && is_movie(&child) {
                    movies.push(child);
                }
            }
        }
    }

    movies.sort();
    Ok(movies)
}

fn frame_total(input: &Path) -> Result<u64, String> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
        ])
        .arg(input)
        .output()
        .map_err(|error| format!("could not start ffprobe: {error}. Is ffmpeg installed?"))?;
    let duration = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("could not determine duration of {}", input.display()))?;
    Ok((duration * 2.0).ceil() as u64)
}

fn saved_frames(output_dir: &Path) -> u64 {
    let mut frame = 1;
    while output_dir.join(format!("frame_{frame:06}.jpg")).is_file() {
        frame += 1;
    }
    frame - 1
}

fn extract_frames(input: &Path, output_dir: &Path, progress: &ProgressBar) -> Result<(), String> {
    fs::create_dir_all(output_dir)
        .map_err(|error| format!("could not create {}: {error}", output_dir.display()))?;

    let total = frame_total(input)?;
    progress.set_length(total);
    let mut saved = saved_frames(output_dir).min(total);
    progress.set_position(saved);
    if saved >= total {
        progress.finish_with_message("already complete");
        return Ok(());
    }
    // The last JPEG may be incomplete if the previous process was interrupted.
    if saved > 0 {
        let _ = fs::remove_file(output_dir.join(format!("frame_{saved:06}.jpg")));
        saved -= 1;
        progress.set_position(saved);
    }

    let start_seconds = saved as f64 / 2.0;
    let mut child = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-ss")
        .arg(format!("{start_seconds:.3}"))
        .arg("-i")
        .arg(input)
        .arg("-vf")
        .arg("fps=2,scale='min(1280,iw)':-2")
        .arg("-q:v")
        .arg("5")
        .arg("-start_number")
        .arg((saved + 1).to_string())
        .arg("-progress")
        .arg("pipe:1")
        .arg("-nostats")
        .arg(output_dir.join("frame_%06d.jpg"))
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|error| format!("could not start ffmpeg: {error}. Is ffmpeg installed?"))?;

    let stdout = child
        .stdout
        .take()
        .ok_or("could not read ffmpeg progress")?;
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        if let Some(value) = line.strip_prefix("frame=")
            && let Ok(frame) = value.parse::<u64>()
        {
            progress.set_position((saved + frame).min(total));
        }
    }
    let status = child
        .wait()
        .map_err(|error| format!("could not wait for ffmpeg: {error}"))?;
    if status.success() {
        progress.set_position(total);
        progress.finish_with_message("complete");
        Ok(())
    } else {
        progress.abandon_with_message("failed; run again to resume");
        Err(format!("ffmpeg failed with status {status}"))
    }
}

fn movie_progress(multi: &MultiProgress, name: &str) -> ProgressBar {
    let bar = multi.add(ProgressBar::new(0));
    bar.set_style(
        ProgressStyle::with_template("{msg:40} {pos}/{len} [{bar:30.cyan/blue}] {eta}")
            .expect("valid progress template"),
    );
    bar.set_message(name.to_owned());
    bar
}

fn run() -> Result<(), String> {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "curtain".to_owned());
    let mut first = args.next();
    if matches!(first.as_deref(), Some("-h" | "--help")) {
        print_usage(&program);
        return Ok(());
    }
    let check = first.as_deref() == Some("--check");
    if check {
        first = args.next();
    }

    let input = first
        .map(PathBuf::from)
        .map_or_else(default_movies_dir, Ok)?;
    let requested_output = args.next().map(PathBuf::from);
    if args.next().is_some() {
        print_usage(&program);
        return Err("too many arguments".to_owned());
    }

    if input.is_file() {
        if check && !confirm(std::slice::from_ref(&input))? {
            println!("Cancelled.");
            return Ok(());
        }
        let output = match requested_output {
            Some(path) => path,
            None => default_output_dir()?.join(
                input
                    .file_stem()
                    .ok_or_else(|| format!("movie has no file name: {}", input.display()))?,
            ),
        };
        let multi = MultiProgress::new();
        let name = input
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("movie");
        let progress = movie_progress(&multi, name);
        return extract_frames(&input, &output, &progress);
    }
    if !input.is_dir() {
        return Err(format!("input does not exist: {}", input.display()));
    }

    let movies = find_movies(&input)?;
    if movies.is_empty() {
        return Err(format!("no supported movies found in {}", input.display()));
    }

    if check && !confirm(&movies)? {
        println!("Cancelled.");
        return Ok(());
    }

    let output_root = match requested_output {
        Some(path) => path,
        None => default_output_dir()?,
    };
    let jobs = env::var("CURTAIN_JOBS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|jobs| *jobs > 0)
        .unwrap_or(2);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .map_err(|error| format!("could not create worker pool: {error}"))?;

    println!("Processing with up to {jobs} movie(s) at once.");
    let multi = Arc::new(MultiProgress::new());
    let completed = multi.add(ProgressBar::new(movies.len() as u64));
    completed.set_style(
        ProgressStyle::with_template("Movies completed {pos}/{len} [{bar:30.green/blue}]")
            .expect("valid progress template"),
    );
    let results: Vec<_> = pool.install(|| {
        movies
            .par_iter()
            .map(|movie| {
                let movie_name = movie
                    .file_stem()
                    .ok_or_else(|| format!("movie has no file name: {}", movie.display()))?;
                let output = output_root.join(movie_name);
                let name = movie_name.to_string_lossy();
                let progress = movie_progress(&multi, &name);
                let result = extract_frames(movie, &output, &progress)
                    .map_err(|error| format!("{}: {error}", movie.display()));
                if result.is_ok() {
                    completed.inc(1);
                }
                result
            })
            .collect()
    });
    completed.finish();

    let failures = results.iter().filter(|result| result.is_err()).count();
    for error in results.into_iter().filter_map(Result::err) {
        eprintln!("error processing {error}");
    }

    println!("Processed {} movie(s); {failures} failed.", movies.len());
    if failures == 0 {
        Ok(())
    } else {
        Err(format!("{failures} movie(s) could not be processed"))
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
