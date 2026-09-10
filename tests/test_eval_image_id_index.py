import unittest

from yolozu.eval.image_id_index import ImageIdIndex


class TestImageIdIndex(unittest.TestCase):
    def test_unique_aliases_remain_portable(self):
        index = ImageIdIndex()
        index.add(r"C:\dataset\scene one.jpg", 0)
        for image in (r"C:\dataset\scene one.jpg", "C:/dataset/scene one.jpg", "scene one.jpg", "scene one", "/moved/scene one.jpg"):
            with self.subTest(image=image):
                self.assertEqual(index.lookup(image), 0)
        self.assertIsNone(index.lookup("unknown.jpg"))

    def test_duplicate_basenames_keep_exact_paths_and_reject_fallback(self):
        for images in (
            [("/dataset/a/sample.jpg", 1), ("/dataset/b/sample.jpg", 2)],
            [("/dataset/b/sample.jpg", 2), ("/dataset/a/sample.jpg", 1)],
        ):
            index = ImageIdIndex()
            for image, image_id in images:
                index.add(image, image_id)
            self.assertEqual(index.lookup("/dataset/a/sample.jpg"), 1)
            self.assertEqual(index.lookup(r"\dataset\b\sample.jpg"), 2)
            self.assertNotIn("sample.jpg", index.as_dict())
            for image in ("sample.jpg", "sample", "/relocated/b/sample.jpg"):
                with self.subTest(image=image, order=images):
                    with self.assertRaisesRegex(ValueError, "ambiguous prediction image key"):
                        index.lookup(image)

    def test_unique_basename_takes_precedence_over_ambiguous_stem(self):
        index = ImageIdIndex()
        index.add("/dataset/sample.jpg", 1)
        index.add("/dataset/sample.png", 2)
        self.assertEqual(index.lookup("sample.jpg"), 1)
        self.assertEqual(index.lookup("sample.png"), 2)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            index.lookup("sample")

    def test_exact_basename_takes_precedence_over_fallback_collision(self):
        index = ImageIdIndex()
        index.add("/dataset/a/sample.jpg", 1)
        index.add("sample.jpg", 2)
        self.assertEqual(index.lookup("sample.jpg"), 2)
        self.assertEqual(index.as_dict()["sample.jpg"], 2)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            index.lookup("/relocated/sample.jpg")

    def test_basename_and_compound_extension_stem_use_separate_namespaces(self):
        for images in (
            [("/dataset/a/sample.jpg", 1), ("/dataset/b/sample.jpg.png", 2)],
            [("/dataset/b/sample.jpg.png", 2), ("/dataset/a/sample.jpg", 1)],
        ):
            index = ImageIdIndex()
            for image, image_id in images:
                index.add(image, image_id)
            for image in ("sample.jpg", "/moved/sample.jpg", r"C:\moved\sample.jpg"):
                with self.subTest(image=image, order=images):
                    self.assertEqual(index.lookup(image), 1)
            self.assertEqual(index.lookup("sample.jpg.png"), 2)
            self.assertEqual(index.lookup("sample"), 1)
            self.assertEqual(index.as_dict()["sample.jpg"], 1)

    def test_bare_compound_extension_stem_remains_available(self):
        index = ImageIdIndex()
        index.add("/dataset/sample.jpg.png", 1)
        self.assertEqual(index.lookup("sample.jpg"), 1)
        self.assertEqual(index.as_dict()["sample.jpg"], 1)

    def test_ambiguous_basename_does_not_fall_back_to_unique_stem(self):
        index = ImageIdIndex()
        index.add("/dataset/a/sample.jpg", 1)
        index.add("/dataset/b/sample.jpg", 2)
        index.add("/dataset/c/sample.jpg.png", 3)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            index.lookup("sample.jpg")
        self.assertNotIn("sample.jpg", index.as_dict())

    def test_duplicate_normalized_dataset_path_is_rejected(self):
        index = ImageIdIndex()
        index.add(r"C:\dataset\sample.jpg", 1)
        with self.assertRaisesRegex(ValueError, "duplicate dataset image key"):
            index.add("C:/dataset/sample.jpg", 2)


if __name__ == "__main__":
    unittest.main()
