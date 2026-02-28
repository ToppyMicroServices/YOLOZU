import unittest

from yolozu.image_keys import add_image_aliases, image_basename, image_key_aliases, lookup_image_alias, require_image_key


class TestImageKeys(unittest.TestCase):
    def test_aliases_include_normalized_and_basename(self):
        aliases = image_key_aliases(r"C:\tmp\foo\0001.jpg")
        self.assertIn(r"C:\tmp\foo\0001.jpg", aliases)
        self.assertIn("C:/tmp/foo/0001.jpg", aliases)
        self.assertIn("0001.jpg", aliases)
        self.assertIn("0001", aliases)

    def test_add_and_lookup_alias(self):
        index: dict[str, int] = {}
        add_image_aliases(index, r"C:\tmp\foo\0001.jpg", 7)
        self.assertEqual(lookup_image_alias(index, "0001.jpg"), 7)
        self.assertEqual(lookup_image_alias(index, "0001"), 7)
        self.assertEqual(lookup_image_alias(index, "C:/tmp/foo/0001.jpg"), 7)

    def test_image_basename_and_require_key(self):
        self.assertEqual(image_basename("/a/b/c.png"), "c.png")
        self.assertEqual(require_image_key(" x.jpg ", where="x"), "x.jpg")
        with self.assertRaises(ValueError):
            require_image_key("", where="x")


if __name__ == "__main__":
    unittest.main()
