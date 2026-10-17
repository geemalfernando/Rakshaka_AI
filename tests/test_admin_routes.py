"""Source-level authorization regression checks (do not need a trained model)."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "src" / "api.py"
DASHBOARD = ROOT / "app" / "dashboard.py"


class AdminRouteSecurityTests(unittest.TestCase):
    def test_sensitive_routes_use_admin_dependency(self):
        tree = ast.parse(API.read_text(encoding="utf-8"))
        protected = {
            "/review_queue", "/review", "/retrain", "/rollback",
            "/predictions", "/alerts", "/review_queue/{rowid}",
        }
        guarded = set()
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) or not decorator.args:
                    continue
                path_arg = decorator.args[0]
                if not isinstance(path_arg, ast.Constant) or path_arg.value not in protected:
                    continue
                dep_keywords = [kw.value for kw in decorator.keywords if kw.arg == "dependencies"]
                found = any(
                    isinstance(dep, ast.Call)
                    and isinstance(dep.func, ast.Name)
                    and dep.func.id == "Depends"
                    and dep.args and isinstance(dep.args[0], ast.Name)
                    and dep.args[0].id == "require_admin"
                    for keyword in dep_keywords if isinstance(keyword, ast.List)
                    for dep in keyword.elts
                )
                self.assertTrue(found, f"{path_arg.value} must require admin authorization")
                guarded.add(path_arg.value)
        self.assertEqual(guarded, protected)

    def test_dashboard_passes_admin_header_to_review_and_retrain(self):
        source = DASHBOARD.read_text(encoding="utf-8")
        self.assertIn('"X-Admin-Token": reviewer_token', source)
        self.assertIn('f"{API}/review", json={"items": items}, headers=headers', source)
        self.assertIn('f"{API}/retrain", headers=headers', source)
        self.assertIn('type="password"', source)

    def test_admin_check_is_fail_closed(self):
        source = API.read_text(encoding="utf-8")
        self.assertIn('if not ADMIN_TOKEN:', source)
        self.assertIn('hmac.compare_digest(x_admin_token, ADMIN_TOKEN)', source)


if __name__ == "__main__":
    unittest.main()
