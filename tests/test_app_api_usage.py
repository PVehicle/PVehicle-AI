"""Kiem tra app.py chi dung cac tham so Streamlit CO THAT.

Vi sao can test nay: cac widget Streamlit khong dong nhat ve ten tham so.
`st.dataframe` va `st.button` nhan `use_container_width`, nhung `st.image`
o ban 1.39 chi nhan `use_column_width` — tham so moi mai den 1.41 moi co.

Loi kieu nay khong lo ra khi import, chi bao khi nguoi dung bam vao dung
chuc nang do. Test nay bat loi ngay tu luc chay kiem thu.

Chay bang: python -m unittest discover tests
"""

import ast
import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

APP_FILE = Path(__file__).resolve().parent.parent / "app.py"

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Cac tham so Streamlit tu nhan qua **kwargs, khong kiem tra duoc bang
# chu ky ham nen bo qua.
SKIP_FUNCTIONS = {"set_page_config", "markdown", "write"}


def collect_streamlit_calls(tree: ast.AST) -> list[tuple[str, str, int]]:
    """Tim cac loi goi st.<ham>(..., <tu_khoa>=...) trong ma nguon.

    Tra ve danh sach (ten_ham, ten_tham_so, so_dong).
    """
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        # Chi quan tam cac loi goi dang `st.<ham>`.
        if not (isinstance(func.value, ast.Name) and func.value.id == "st"):
            continue

        for keyword in node.keywords:
            # **kwargs khong co ten, bo qua.
            if keyword.arg is not None:
                calls.append((func.attr, keyword.arg, node.lineno))
    return calls


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Chua cai streamlit")
class TestStreamlitApiUsage(unittest.TestCase):
    """Moi tham so truyen vao Streamlit phai ton tai that."""

    @classmethod
    def setUpClass(cls):
        source = APP_FILE.read_text(encoding="utf-8")
        cls.calls = collect_streamlit_calls(ast.parse(source))

    def test_tim_duoc_loi_goi_streamlit(self):
        """Neu khong tim thay loi goi nao thi test sau vo nghia."""
        self.assertGreater(len(self.calls), 5)

    def test_moi_tham_so_deu_ton_tai(self):
        problems = []

        for func_name, keyword, lineno in self.calls:
            if func_name in SKIP_FUNCTIONS:
                continue

            func = getattr(st, func_name, None)
            if func is None:
                problems.append(
                    f"app.py:{lineno} — st.{func_name} khong ton tai"
                )
                continue

            try:
                params = inspect.signature(func).parameters
            except (ValueError, TypeError):
                # Mot so ham duoc boc khien khong doc duoc chu ky.
                continue

            # Ham nhan **kwargs thi khong kiem tra duoc.
            has_var_keyword = any(
                p.kind is inspect.Parameter.VAR_KEYWORD
                for p in params.values()
            )
            if has_var_keyword:
                continue

            if keyword not in params:
                problems.append(
                    f"app.py:{lineno} — st.{func_name}() khong nhan "
                    f"tham so {keyword!r} (ban {st.__version__})"
                )

        self.assertEqual(
            problems, [],
            "Tham so Streamlit khong hop le:\n  "
            + "\n  ".join(problems),
        )

    def test_st_image_dung_dung_ten_tham_so(self):
        """Bat rieng loi da tung gap: st.image dung use_container_width."""
        image_keywords = {
            keyword for name, keyword, _ in self.calls if name == "image"
        }
        params = inspect.signature(st.image).parameters

        if "use_container_width" not in params:
            self.assertNotIn(
                "use_container_width", image_keywords,
                f"Streamlit {st.__version__}: st.image dung "
                "`use_column_width`, khong phai `use_container_width`.",
            )


class TestKhongLongExpander(unittest.TestCase):
    """Streamlit cam long expander trong expander.

    Loi nay chi bao khi nguoi dung mo dung nhanh giao dien do:
      StreamlitAPIException: Expanders may not be nested inside other
      expanders.

    Kiem tra bang cach lan theo loi goi ham: neu mot ham chua expander
    va duoc goi tu ben trong mot expander khac, do la loi.
    """

    @classmethod
    def setUpClass(cls):
        source = APP_FILE.read_text(encoding="utf-8")
        cls.tree = ast.parse(source)

    @staticmethod
    def _expander_nodes(node: ast.AST) -> list[ast.With]:
        """Tim cac khoi `with st.expander(...)` trong mot nhanh cay."""
        found = []
        for child in ast.walk(node):
            if not isinstance(child, ast.With):
                continue
            for item in child.items:
                call = item.context_expr
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "expander"
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "st"
                ):
                    found.append(child)
        return found

    def _functions_with_expander(self) -> set[str]:
        """Ten cac ham co chua st.expander."""
        names = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                if self._expander_nodes(node):
                    names.add(node.name)
        return names

    def test_khong_goi_ham_chua_expander_tu_trong_expander(self):
        producers = self._functions_with_expander()
        problems = []

        for node in ast.walk(self.tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for expander in self._expander_nodes(node):
                # Trong than cua expander nay co goi ham nao cung tao
                # expander khong?
                for inner in ast.walk(expander):
                    if not isinstance(inner, ast.Call):
                        continue
                    if isinstance(inner.func, ast.Name):
                        called = inner.func.id
                    elif isinstance(inner.func, ast.Attribute):
                        called = inner.func.attr
                    else:
                        continue
                    if called in producers:
                        problems.append(
                            f"app.py:{inner.lineno} — goi {called}() "
                            "tu trong st.expander, ma ham do cung tao "
                            "expander (Streamlit cam long nhau)"
                        )

        self.assertEqual(
            problems, [],
            "Long expander trong expander:\n  " + "\n  ".join(problems),
        )

    def test_khong_long_expander_truc_tiep(self):
        """Truong hop hien nhien: expander viet long ngay trong nhau."""
        problems = []
        for outer in self._expander_nodes(self.tree):
            for child in ast.iter_child_nodes(outer):
                for inner in self._expander_nodes(child):
                    if inner is not outer:
                        problems.append(f"app.py:{inner.lineno}")

        self.assertEqual(problems, [], f"Expander long nhau: {problems}")


if __name__ == "__main__":
    unittest.main()
