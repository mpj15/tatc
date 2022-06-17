import unittest

from tatc.schemas import TwoLineElements
from pydantic import ValidationError
from datetime import datetime, timezone
import numpy as np


class TestTLE(unittest.TestCase):
    def test_good_data(self):
        good_data = {
            "tle": [
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        }
        o = TwoLineElements(**good_data)
        self.assertEqual(o.tle[0], good_data.get("tle")[0])
        self.assertEqual(o.tle[1], good_data.get("tle")[1])

    def test_bad_data_invalid(self):
        bad_data = {"tle": ["not valid", "not valid"]}
        with self.assertRaises(ValidationError):
            TwoLineElements(**bad_data)

    def test_bad_data_checksums(self):
        bad_data = {
            "tle": [
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9994",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        }
        with self.assertRaises(ValidationError):
            TwoLineElements(**bad_data)

    def test_get_catalog_number(self):
        o = TwoLineElements(
            tle=[
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        )
        self.assertEqual(o.get_catalog_number(), 25544)

    def test_get_classification(self):
        o = TwoLineElements(
            tle=[
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        )
        self.assertEqual(o.get_classification(), "U")

    def test_get_international_designator(self):
        o = TwoLineElements(
            tle=[
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        )
        self.assertEqual(o.get_international_designator(), "1998-067A")

    def test_get_epoch(self):
        o = TwoLineElements(
            tle=[
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        )
        self.assertEqual(
            o.get_epoch(), datetime(2021, 6, 6, 7, 19, 36, 128928, tzinfo=timezone.utc)
        )

    def test_get_first_derivative_mean_motion(self):
        o = TwoLineElements(
            tle=[
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        )
        self.assertEqual(o.get_first_derivative_mean_motion(), 0.00003432)

    def test_get_second_derivative_mean_motion(self):
        o = TwoLineElements(
            tle=[
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        )
        self.assertEqual(o.get_second_derivative_mean_motion(), 0.0)

    def test_get_inclination(self):
        o = TwoLineElements(
            tle=[
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        )
        self.assertEqual(o.get_inclination(), 51.6455)

    def test_get_right_ascension_ascending_node(self):
        o = TwoLineElements(
            tle=[
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        )
        self.assertEqual(o.get_right_ascension_ascending_node(), 41.4969)

    def test_get_eccentricity(self):
        o = TwoLineElements(
            tle=[
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        )
        self.assertEqual(o.get_eccentricity(), 0.0003508)

    def test_get_perigee_argument(self):
        o = TwoLineElements(
            tle=[
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        )
        self.assertEqual(o.get_perigee_argument(), 68.0432)

    def test_get_mean_anomaly(self):
        o = TwoLineElements(
            tle=[
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        )
        self.assertEqual(o.get_mean_anomaly(), 78.3395)

    def test_get_mean_motion(self):
        o = TwoLineElements(
            tle=[
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        )
        self.assertAlmostEqual(o.get_mean_motion(), 15.48957534)

    def test_get_revolution_number_at_epoch(self):
        o = TwoLineElements(
            tle=[
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        )
        self.assertEqual(o.get_revolution_number_at_epoch(), 28675)

    def test_get_semimajor_axis(self):
        o = TwoLineElements(
            tle=[
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        )
        self.assertAlmostEqual(o.get_semimajor_axis(), 6797911, delta=1.0)

    def test_get_altitude(self):
        o = TwoLineElements(
            tle=[
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        )
        self.assertAlmostEqual(o.get_altitude(), 426902, delta=1.0)

    def test_get_true_anomaly(self):
        o = TwoLineElements(
            tle=[
                "1 25544U 98067A   21156.30527927  .00003432  00000-0  70541-4 0  9993",
                "2 25544  51.6455  41.4969 0003508  68.0432  78.3395 15.48957534286754",
            ]
        )
        self.assertAlmostEqual(o.get_true_anomaly(), 78.3788725993742)
