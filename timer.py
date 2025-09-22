import time
from contextlib import contextmanager

class SimpleTimer:
    """一個簡單的計時器類別，用於測量程式碼區塊的執行時間。"""
    def __init__(self):
        self.totals = {}
        self.counts = {}
        self.start_times = {}

    def start(self, name: str):
        """開始一個命名計時器。"""
        self.start_times[name] = time.perf_counter()

    def stop(self, name: str):
        """停止一個命名計時器並累加時間。"""
        if name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[name]
            self.totals[name] = self.totals.get(name, 0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1
            del self.start_times[name]
            return elapsed
        return 0

    @contextmanager
    def record(self, name: str):
        """使用 'with' 語句來自動計時。"""
        self.start(name)
        yield
        self.stop(name)

    def report(self):
        """打印所有計時器的統計報告。"""
        print("\n--- 計時器分析報告 ---")
        if not self.totals:
            print("沒有任何計時記錄。")
            return
            
        # 按照總耗時排序
        sorted_items = sorted(self.totals.items(), key=lambda item: item[1], reverse=True)
        
        for name, total_time in sorted_items:
            count = self.counts[name]
            avg_time = total_time / count
            print(f"[{name}]:")
            print(f"  - 總耗時: {total_time:.4f} 秒")
            print(f"  - 執行次數: {count} 次")
            print(f"  - 平均耗時: {avg_time:.4f} 秒/次")
        print("------------------------\n")