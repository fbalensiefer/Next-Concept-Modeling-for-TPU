import io
import json

from ncm.engine_rpc import response, serve


def test_ping_is_jsonrpc_and_unknown_method_is_method_error():
    assert response({"jsonrpc": "2.0", "id": 9, "method": "engine.ping"}) == {
        "jsonrpc": "2.0", "id": 9, "result": {"status": "ready"}}
    error = response({"jsonrpc": "2.0", "id": "x", "method": "no.such.method"})
    assert error["error"]["code"] == -32601 and error["id"] == "x"


def test_stdio_is_one_json_response_per_input_line_including_parse_errors():
    source = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"engine.ping"}\nnot json\n')
    sink = io.StringIO()
    serve(source, sink)
    rows = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert [row["id"] for row in rows] == [1, None]
    assert rows[0]["result"] == {"status": "ready"}
    assert rows[1]["error"]["code"] == -32700
