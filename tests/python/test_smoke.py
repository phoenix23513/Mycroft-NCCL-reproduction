import unittest

import mycroft


class PackageSmokeTest(unittest.TestCase):
    def test_package_is_importable(self) -> None:
        self.assertEqual(mycroft.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
