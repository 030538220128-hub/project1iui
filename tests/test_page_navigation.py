"""Regression checks for model caches shared across Streamlit pages."""

from pathlib import Path
import unittest

import streamlit as st
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
PRICE = "pages/1_House_Price_Prediction.py"
ANOMALY = "pages/2_Anomaly_Detection.py"


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
            else:
                app.radio[0].set_value("✍️ Kiểm tra tin mới").run()
                for field in app.selectbox:
                    if field.label in ("Quận", "Loại hình"):
                        field.select(field.options[1])
                app.button[0].click().run()
                self.assertIn("Composite Score", [m.label for m in app.metric])
            self.assertEqual([e.message for e in app.exception], [], page)
            self.assertEqual([e.value for e in app.error], [], page)

    def test_price_then_anomaly_then_price(self):
        self.check_navigation([PRICE, ANOMALY, PRICE])

    def test_anomaly_then_price_then_anomaly(self):
        self.check_navigation([ANOMALY, PRICE, ANOMALY])


if __name__ == "__main__":
    unittest.main()
