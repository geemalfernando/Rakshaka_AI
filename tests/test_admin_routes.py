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
            "/monitor", "/simulate", "/simulate/reset",
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

    def test_dashboard_passes_admin_header_to_every_admin_call(self):
        """Every requests.<method>(f"{API}/<admin path>...") call in the dashboard must send headers=."""
        source = DASHBOARD.read_text(encoding="utf-8")
        tree = ast.parse(source)
        admin_paths = ("/review_queue", "/review", "/retrain", "/rollback", "/predictions", "/alerts", "/monitor", "/simulate")
        seen = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "requests"
                    and node.args and isinstance(node.args[0], ast.JoinedStr)):
                continue
            literal = "".join(v.value for v in node.args[0].values if isinstance(v, ast.Constant))
            path = next((p for p in admin_paths if literal.startswith(p)), None)
            if path is None:
                continue
            seen.add(path)
            self.assertIn("headers", [kw.arg for kw in node.keywords],
                          f"requests.{node.func.attr}(...{literal}) must send the admin token header")
        self.assertEqual(seen, set(admin_paths), "every admin endpoint should be reachable from the dashboard")
        self.assertIn('"X-Admin-Token"', source)
        self.assertIn('type="password"', source)

    def test_dashboard_never_simulates_through_public_predict(self):
        source = DASHBOARD.read_text(encoding="utf-8")
        self.assertNotIn('requests.post(f"{API}/predict", json={"text": text, "language": language}', source)
        self.assertIn('f"{API}/simulate"', source)

    def test_dashboard_holds_no_hard_coded_admin_secret(self):
        source = DASHBOARD.read_text(encoding="utf-8")
        self.assertNotIn("ADMIN_PASSWORD", source)
        self.assertNotIn("scamproof", source)

    def test_admin_check_is_fail_closed(self):
        source = API.read_text(encoding="utf-8")
        self.assertIn('if not ADMIN_TOKEN:', source)
        self.assertIn('hmac.compare_digest(x_admin_token, ADMIN_TOKEN)', source)


if __name__ == "__main__":
    unittest.main()
