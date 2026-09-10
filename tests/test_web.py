import unittest
from types import SimpleNamespace
from apps import server


class GalleryFilters(unittest.TestCase):
    def setUp(self):
        self.previous = getattr(server, 'engine', None)
        movies = [dict(id='Train', title='Train', training_status='training', training_label='Training', start=0, frames=3),
                  dict(id='Validate', title='Validate', training_status='validation', training_label='Validation', start=3, frames=3)]
        records = [dict(movie=m['id'], filename=f'frame_{i}.jpg', frame_number=i)
                   for m in movies for i in range(3)]
        server.engine = SimpleNamespace(movies=movies, movie_lookup={m['id']:m for m in movies}, records=records)
        self.client = server.app.test_client()

    def tearDown(self):
        server.engine = self.previous

    def test_groups(self):
        for group, count in [('all',4), ('training',2), ('validation',2)]:
            r = self.client.get('/gallery?group='+group)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.data.count(b'<figure>'), count)
            self.assertIn(f'value="{group}" selected'.encode(), r.data)
        self.assertNotIn(b'[Validation]', self.client.get('/gallery?group=training').data)
        self.assertNotIn(b'[Training]', self.client.get('/gallery?group=validation').data)

    def test_invalid_and_empty(self):
        self.assertIn(b'No images in this group', self.client.get('/gallery?group=unseen').data)
        self.assertEqual(self.client.get('/gallery?group=bad').status_code, 400)
        self.assertEqual(self.client.get('/gallery?movie=missing').status_code, 404)
        self.assertIn(b'No images in this group', self.client.get('/gallery?movie=Train&group=validation').data)
        self.assertEqual(self.client.get('/gallery?movie=Train').data.count(b'<figure>'),3)

    def test_noncontiguous_spans(self):
        movie = server.engine.movies[0]
        movie['frames'] = 6
        movie['spans'] = [(0,3),(6,3)]
        server.engine.records.extend([dict(movie='Train', filename=f'extra_{i}.jpg', frame_number=i) for i in range(3)])
        response = self.client.get('/gallery?movie=Train')
        self.assertEqual(response.status_code,200)
        self.assertIn(b'/frame/6',response.data)
        self.assertNotIn(b'[Validation]',response.data)


if __name__ == '__main__':
    unittest.main()
