"""File-input unit tests runnable without IceNet's scientific dependencies.

From the repository root:
    python -B -m unittest discover -s tests -p test_cli_file_input.py -v
"""
import argparse
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from icenet.utils import read_text_arg


class BrokenInput(io.StringIO):
    def read(self, *args, **kwargs):
        raise OSError("input read failed")


class ReadTextArgTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="icenet-cli-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "input [final] $value & text.txt"
        self.parser = argparse.ArgumentParser(prog="icenet-test")

    def test_file_content_and_close(self):
        text = "2026-01-01\nTexto con \u00f1\n"
        with self.path.open("w") as stream:
            stream.write(text)
        opened = []
        real_open = open

        def track_open(*args, **kwargs):
            stream = real_open(*args, **kwargs)
            opened.append(stream)
            return stream

        with mock.patch("icenet.utils.open", side_effect=track_open, create=True):
            self.assertEqual(read_text_arg(self.parser, str(self.path)), text)
        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0].closed)

    def test_empty_file(self):
        self.path.write_bytes(b"")
        self.assertEqual(read_text_arg(self.parser, str(self.path)), "")

    def test_missing_file_reports_parser_error(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            read_text_arg(self.parser, str(self.path))
        self.assertEqual(error.exception.code, 2)
        self.assertIn(self.path.name, stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_permission_error_reports_parser_error(self):
        stderr = io.StringIO()
        with mock.patch("icenet.utils.open", side_effect=PermissionError("unreadable"), create=True):
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                read_text_arg(self.parser, str(self.path))
        self.assertEqual(error.exception.code, 2)
        self.assertIn(self.path.name, stderr.getvalue())

    def test_owned_stream_closes_after_read_failure(self):
        stream = BrokenInput()
        with mock.patch("icenet.utils.open", return_value=stream, create=True):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                read_text_arg(self.parser, str(self.path))
        self.assertEqual(error.exception.code, 2)
        self.assertTrue(stream.closed)

    def test_stdin_is_borrowed(self):
        with io.StringIO("2026-01-01\n") as stream:
            with mock.patch.object(sys, "stdin", stream):
                self.assertEqual(read_text_arg(self.parser, "-"), "2026-01-01\n")
            self.assertFalse(stream.closed)

    def test_stdin_stays_open_after_read_failure(self):
        with BrokenInput() as stream:
            with mock.patch.object(sys, "stdin", stream):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    read_text_arg(self.parser, "-")
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(stream.closed)


if __name__ == "__main__":
    unittest.main()
