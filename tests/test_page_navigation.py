"""Regression checks for model caches, 19-feature parity, and page navigation in project1iui."""

from pathlib import Path
import unittest

import streamlit as st
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
PRICE = "pages/1_House_Price_Prediction.py"
ANOMALY = "pages/2_Anomaly_Detection.py"
OVERVIEW = "pages/3_Market_Overview.py"


class PageNavigationTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()

    def tearDown(self):
        st.cache_resource.clear()

    def check_navigation(self, pages):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(list(app.warning), [])
        for page in pages:
            app.switch_page(page).run()
            self.assertEqual([e.message for e in app.exception], [], page)
            self.assertEqual([e.value for e in app.error], [], page)
            if page == PRICE:
                app.button[0].click().run()
                price_metrics = {m.label: m.value for m in app.metric}
                self.assertIn("Giá dự đoán", price_metrics)
                self.assertNotEqual(price_metrics["Giá dự đoán"], "0.00 tỷ VNĐ")
            elif page == ANOMALY:
                app.radio[0].set_value("✍️ Kiểm tra tin mới").run()
                for field in app.selectbox:
                    if field.label in ("Quận", "Loại hình"):
                        field.select(field.options[1])
                app.button[0].click().run()
                self.assertIn("Composite Score", [m.label for m in app.metric])
            elif page == OVERVIEW:
                self.assertGreaterEqual(len(app.metric), 4)
            self.assertEqual([e.message for e in app.exception], [], page)
            self.assertEqual([e.value for e in app.error], [], page)

    def test_price_then_anomaly_then_overview(self):
        self.check_navigation([PRICE, ANOMALY, OVERVIEW, PRICE])

    def test_anomaly_then_price_then_anomaly(self):
        self.check_navigation([ANOMALY, PRICE, ANOMALY])

    def test_price_and_anomaly_predict_identical_price_for_same_house(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

        app.switch_page(PRICE).run()
        for field in app.selectbox:
            if field.label in ("Quận", "Loại hình"):
                field.select(field.options[1])
        app.button[0].click().run()
        p1_val = {m.label: m.value for m in app.metric}["Giá dự đoán"].replace(" VNĐ", "")

        app.switch_page(ANOMALY).run()
        app.radio[0].set_value("✍️ Kiểm tra tin mới").run()
        for field in app.selectbox:
            if field.label in ("Quận", "Loại hình"):
                field.select(field.options[1])
        app.button[0].click().run()
        p2_val = {m.label: m.value for m in app.metric}["Giá model kỳ vọng"]

        self.assertEqual(p1_val, p2_val)


if __name__ == "__main__":
    unittest.main()
