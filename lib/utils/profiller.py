import sys
import time
from collections import defaultdict

class FunctionProfiler:
    def __init__(self, module_path_filter=None, min_pct=0.5):
        """
        min_pct: hide nodes that are less than this % of their parent's time
                 (set to 0 to show everything)
        """
        self.filter  = module_path_filter
        self.min_pct = min_pct
        self.stack   = []
        self.roots   = []

        # Names to always suppress from the tree
        self._skip_names = {
            '<genexpr>', '<listcomp>', '<dictcomp>', '<setcomp>',
            '<lambda>', '<module>'
        }

    def __enter__(self):
        sys.settrace(self.trace)
        return self

    def trace(self, frame, event, arg):
        if not self.is_mine(frame):
            return self.trace

        func = frame.f_code.co_name

        # Skip noise names entirely
        if func in self._skip_names:
            return self.trace

        path = frame.f_code.co_filename.split('/')[-1]

        if event == 'call':
            node = {
                'name':     func,
                'file':     path,
                'start':    time.perf_counter(),
                'children': [],
                'elapsed':  None,
            }
            if self.stack:
                self.stack[-1]['children'].append(node)
            else:
                self.roots.append(node)
            self.stack.append(node)

        elif event == 'return':
            if self.stack and self.stack[-1]['name'] == func:
                node = self.stack.pop()
                node['elapsed'] = time.perf_counter() - node['start']

        return self.trace

    def is_mine(self, frame):
        path = frame.f_code.co_filename
        noise = ['site-packages', '<frozen', 'importlib', 'torch/',
                 'wandb/', 'pydantic', 'requests', 'transformers/']
        if any(n in path for n in noise):
            return False
        if self.filter:
            return self.filter in path
        return True

    def __exit__(self, *args):
        sys.settrace(None)

    def print_tree(self, nodes=None, indent=0, parent_time=None, file=None):
        if nodes is None:
            nodes = self.aggregate(self.roots)
            print(f"\n{'Function':<48} {'Calls':>6}  {'Total':>10}  {'Avg':>10}  {'% parent':>9}  Source", file=file)
            print("─" * 100, file=file)
            parent_time = sum(n['elapsed'] for n in nodes if n['elapsed'])

        for node in sorted(nodes, key=lambda x: x['elapsed'], reverse=True):
            t     = node['elapsed']
            calls = node['calls']
            pct   = (100 * t / parent_time) if parent_time else 0

            # Skip trivial nodes
            if parent_time and pct < self.min_pct:
                continue

            avg    = t / calls if calls else 0
            prefix = "  " * indent + ("└─ " if indent else "")
            label  = prefix + node['name']
            avg_str = f"{avg*1000:8.1f}ms" if calls > 1 else "          "

            print(f"{label:<48}  {calls:>6}  {t*1000:>8.1f}ms  {avg_str}  {pct:>7.1f}%  [{node['file']}]", file=file)

            self.print_tree(node['children'], indent + 1, parent_time=t, file=file)

    def aggregate(self, nodes):
        """
        Collapse repeated calls to the same function into one summary node.
        e.g. 130x generate_response → one row showing total + avg + count.
        Children are merged recursively.
        """
        seen   = {}   # name+file → aggregated node
        order  = []   # preserve first-seen order

        for node in nodes:
            if node['elapsed'] is None:
                continue
            key = (node['name'], node['file'])
            if key not in seen:
                seen[key] = {
                    'name':     node['name'],
                    'file':     node['file'],
                    'elapsed':  0.0,
                    'calls':    0,
                    'children': [],
                }
                order.append(key)
            agg = seen[key]
            agg['elapsed'] += node['elapsed']
            agg['calls']   += 1
            agg['children'].extend(node['children'])

        # Recursively aggregate children of each merged node
        result = []
        for key in order:
            agg = seen[key]
            agg['children'] = self.aggregate(agg['children'])
            result.append(agg)

        return result