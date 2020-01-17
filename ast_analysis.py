import ast, sys, re, os
from pprint import pprint as pp


def main():
    analysis_file = open("ast_report.txt","w")
    for path, dirs, files in os.walk('dask'):
        for file in files:
            if file.endswith('.py'):
                source = open(os.path.join(path,file), "r")
                tree = ast.parse(source.read())
                analyzer = Analyzer()
                analyzer.visit(tree)
                codelist=analyzer.report()
                analysis_file.write('\n' + os.path.join(path,file) + '\n\n')
                [analysis_file.write(line+'\n') for line in codelist]
    analysis_file.close()

class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = []
        self.depth=0

    def visit_FunctionDef(self, node):
        self.depth = self.depth + 4
        depths = '-'*self.depth
        self.stats.append(depths + ' Func: ' + node.name)
        self.generic_visit(node)
        self.depth = self.depth - 4
    def visit_ClassDef(self, node):
        self.depth = self.depth + 4
        depths = '-'*self.depth 
        self.stats.append(depths + ' Class: ' + node.name)
        self.generic_visit(node)
        self.depth = self.depth - 4
        

    def report(self):
        return self.stats

if __name__ == "__main__":
    main()
