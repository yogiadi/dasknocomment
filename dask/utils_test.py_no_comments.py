def inc(x):
    return x + 1
def dec(x):
    return x - 1
def add(x, y):
    return x + y
class GetFunctionTestMixin(object):
    
    def test_get(self):
        d = {":x": 1, ":y": (inc, ":x"), ":z": (add, ":x", ":y")}
        assert self.get(d, ":x") == 1
        assert self.get(d, ":y") == 2
        assert self.get(d, ":z") == 3
    def test_badkey(self):
        d = {":x": 1, ":y": (inc, ":x"), ":z": (add, ":x", ":y")}
        try:
            result = self.get(d, "badkey")
        except KeyError:
            pass
        else:
            msg = "Expected `{}` with badkey to raise KeyError.\n"
            msg += "Obtained '{}' instead.".format(result)
            assert False, msg.format(self.get.__name__)
    def test_nested_badkey(self):
        d = {"x": 1, "y": 2, "z": (sum, ["x", "y"])}
        try:
            result = self.get(d, [["badkey"], "y"])
        except KeyError:
            pass
        else:
            msg = "Expected `{}` with badkey to raise KeyError.\n"
            msg += "Obtained '{}' instead.".format(result)
            assert False, msg.format(self.get.__name__)
    def test_data_not_in_dict_is_ok(self):
        d = {"x": 1, "y": (add, "x", 10)}
        assert self.get(d, "y") == 11
    def test_get_with_list(self):
        d = {"x": 1, "y": 2, "z": (sum, ["x", "y"])}
        assert self.get(d, ["x", "y"]) == (1, 2)
        assert self.get(d, "z") == 3
    def test_get_with_list_top_level(self):
        d = {
            "a": [1, 2, 3],
            "b": "a",
            "c": [1, (inc, 1)],
            "d": [(sum, "a")],
            "e": ["a", "b"],
            "f": [[[(sum, "a"), "c"], (sum, "b")], 2],
        }
        assert self.get(d, "a") == [1, 2, 3]
        assert self.get(d, "b") == [1, 2, 3]
        assert self.get(d, "c") == [1, 2]
        assert self.get(d, "d") == [6]
        assert self.get(d, "e") == [[1, 2, 3], [1, 2, 3]]
        assert self.get(d, "f") == [[[6, [1, 2]], 6], 2]
    def test_get_with_nested_list(self):
        d = {"x": 1, "y": 2, "z": (sum, ["x", "y"])}
        assert self.get(d, [["x"], "y"]) == ((1,), 2)
        assert self.get(d, "z") == 3
    def test_get_works_with_unhashables_in_values(self):
        f = lambda x, y: x + len(y)
        d = {"x": 1, "y": (f, "x", set([1]))}
        assert self.get(d, "y") == 2
    def test_nested_tasks(self):
        d = {"x": 1, "y": (inc, "x"), "z": (add, (inc, "x"), "y")}
        assert self.get(d, "z") == 4
    def test_get_stack_limit(self):
        d = {"x%d" % (i + 1): (inc, "x%d" % i) for i in range(10000)}
        d["x0"] = 0
        assert self.get(d, "x10000") == 10000
    def test_with_HighLevelGraph(self):
        from .highlevelgraph import HighLevelGraph
        layers = {"a": {"x": 1, "y": (inc, "x")}, "b": {"z": (add, (inc, "x"), "y")}}
        dependencies = {"a": (), "b": {"a"}}
        graph = HighLevelGraph(layers, dependencies)
        assert self.get(graph, "z") == 4
