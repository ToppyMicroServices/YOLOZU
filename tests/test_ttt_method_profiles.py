import unittest

from yolozu.tta.method_profiles import TTT_METHOD_PROFILES, get_ttt_method_profile


class TestTTTMethodProfiles(unittest.TestCase):
    def test_profiles_make_fidelity_and_detector_semantics_explicit(self):
        self.assertEqual(
            set(TTT_METHOD_PROFILES), {"tent", "mim", "cotta", "eata", "sar"}
        )
        self.assertEqual(
            TTT_METHOD_PROFILES["tent"]["implementation_class"], "detector_adapted"
        )
        self.assertEqual(
            TTT_METHOD_PROFILES["mim"]["implementation_class"], "conditional_model_hook"
        )
        self.assertEqual(
            TTT_METHOD_PROFILES["mim"]["profile_id"], "yolozu_structured_mim_v1"
        )
        for method in ("cotta", "eata", "sar"):
            profile = TTT_METHOD_PROFILES[method]
            self.assertEqual(profile["profile_id"], "yolozu_phase1_variant")
            self.assertFalse(profile["reference_faithful"])
            self.assertEqual(profile["efficacy"], "not_established")
        for profile in TTT_METHOD_PROFILES.values():
            self.assertIs(profile["runnable"], True)
            self.assertEqual(profile["maturity"], "research")
            self.assertIn("fidelity", profile)
            self.assertEqual(profile["efficacy"], "not_established")
            semantics = profile["loss"]["detector_logits"]
            self.assertEqual(semantics["class_axis"], "last")
            self.assertIn("queries", semantics["query_semantics"])
            self.assertEqual(semantics["foreground_selection"], "none")
            self.assertEqual(semantics["query_correspondence"], "not_established")
            self.assertIn("included_if_present", semantics["no_object_semantics"])
        self.assertEqual(
            TTT_METHOD_PROFILES["cotta"]["loss"]["augmented_view_query_correspondence"],
            "not_established",
        )

    def test_profile_return_isolated_from_global_registry(self):
        profile = get_ttt_method_profile("tent")
        profile["loss"]["detector_logits"]["class_axis"] = "mutated"
        self.assertEqual(
            TTT_METHOD_PROFILES["tent"]["loss"]["detector_logits"]["class_axis"],
            "last",
        )

    def test_unknown_profile_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown TTT method profile"):
            get_ttt_method_profile("unknown")


if __name__ == "__main__":
    unittest.main()
