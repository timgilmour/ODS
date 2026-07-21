from app.gateway import detect_default_gateway


def _write(tmp_path, body):
    p = tmp_path / "route"
    p.write_text("Iface\tDestination\tGateway\tFlags\n" + body, encoding="utf-8")
    return str(p)


def test_parses_little_endian_gateway(tmp_path):
    # 0112A8C0 little-endian -> 192.168.18.1
    path = _write(tmp_path, "eth0\t00000000\t0112A8C0\t0003\n")
    assert detect_default_gateway(path) == "192.168.18.1"


def test_skips_non_default_and_non_gateway_rows(tmp_path):
    body = "eth0\t0000FEA9\t00000000\t0001\neth0\t00000000\t00000000\t0001\n"
    assert detect_default_gateway(_write(tmp_path, body)) == ""


def test_missing_file_returns_empty():
    assert detect_default_gateway("/nonexistent/route") == ""
