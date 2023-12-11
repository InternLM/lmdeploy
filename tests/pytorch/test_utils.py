from lmdeploy.legacy.pytorch.utils import BasicStreamer, TerminalIO


def test_terminal_io(monkeypatch):
    import io
    tio = TerminalIO()
    inputs = 'hello\n\n'
    # inputs = 'hello\n\n\x1B[A\n\n'
    monkeypatch.setattr('sys.stdin', io.StringIO(inputs))
    string = tio.input()
    tio.output(string)
    assert string == 'hello'
    # string = tio.input()
    # tio.output(string)
    # assert string == 'hello'


def test_basic_streamer():
    output = []

    def decode_func(value):
        return value + 10

    def output_func(value):
        output.append(value)

    streamer = BasicStreamer(decode_func, output_func)
    for i in range(10):
        streamer.put(i)
        if i == 5:
            streamer.end()
    streamer.end()

    assert output == [11, 12, 13, 14, 15, '\n', 17, 18, 19, '\n']

    output.clear()
    streamer = BasicStreamer(decode_func, output_func, skip_prompt=False)
    for i in range(10):
        streamer.put(i)
        if i == 5:
            streamer.end()
    streamer.end()

    assert output == [10, 11, 12, 13, 14, 15, '\n', 16, 17, 18, 19, '\n']
