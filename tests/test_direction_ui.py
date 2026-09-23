import threading
import unittest
from unittest.mock import Mock, patch
from PIL import Image

from app import build_app


class DirectionUiTests(unittest.TestCase):
    def test_changing_input_discards_running_direction_result(self):
        app = build_app(Mock())
        callbacks = {f.fn.__name__:f.fn for f in app.fns.values() if f.fn}
        self.assertIn('parse_directions',callbacks)
        state = {'revision':0}
        entered, release = threading.Event(), threading.Event()
        answers = []
        def slow_read(*args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test failed to release inference')
            return {}
        with patch('app.read_directions',side_effect=slow_read), \
             patch('app.draw_evidence'), patch('app.summarize_directions',return_value='old result'):
            thread = threading.Thread(target=lambda:answers.append(callbacks['parse_directions'](
                Image.new('RGB',(300,300)), .25, state)))
            thread.start()
            self.assertTrue(entered.wait(5))
            cleared = callbacks['invalidate_directions'](state)
            self.assertEqual(cleared[:3], ('',None,{}))
            release.set();thread.join(5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(answers, [({'__type__':'update'},{'__type__':'update'},{'__type__':'update'})])


if __name__=='__main__':unittest.main()
