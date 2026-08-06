import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class RoutingDecision:
    sample_id: str
    cluster_id: Optional[str]
    no_retrieval: bool
    passed_chunks: int
    reason: str


class NoRetrievalRouter:
    def __init__(self, log_path: Path, min_passed_chunks: int = 1):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_passed_chunks = max(1, min_passed_chunks)
        self._decisions: List[RoutingDecision] = []

    def route_sample(self, sample_id: str, cluster_id: Optional[str], passed_chunks: int) -> RoutingDecision:
        """Route sample to no_retrieval track if passed chunks are fewer than min_passed_chunks."""
        no_retrieval = passed_chunks < self.min_passed_chunks
        reason = (
            f"Insufficient retrieved chunks: {passed_chunks} passed threshold (minimum required is {self.min_passed_chunks})"
            if no_retrieval
            else "Sufficient context retrieved"
        )

        decision = RoutingDecision(
            sample_id=sample_id,
            cluster_id=cluster_id,
            no_retrieval=no_retrieval,
            passed_chunks=passed_chunks,
            reason=reason
        )

        self._decisions.append(decision)
        return decision

    def flush_batch(self) -> None:
        """Append decisions currently in buffer to the log file."""
        if not self._decisions:
            return

        with open(self.log_path, "a", encoding="utf-8") as f:
            for decision in self._decisions:
                f.write(json.dumps(asdict(decision)) + "\n")
        self._decisions.clear()

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Compute aggregate no-retrieval stats per cluster and overall."""
        # Read the file to get all logged decisions (since flush_batch might be called multiple times)
        all_decisions = []
        if self.log_path.exists():
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_decisions.append(json.loads(line))

        if not all_decisions:
            return {"overall_rate": 0.0, "total_samples": 0, "no_retrieval_count": 0, "by_cluster": {}}

        total_samples = len(all_decisions)
        no_retrieval_count = sum(1 for d in all_decisions if d["no_retrieval"])
        overall_rate = no_retrieval_count / total_samples

        cluster_stats = {}
        for d in all_decisions:
            cluster = d.get("cluster_id") or "unknown"
            if cluster not in cluster_stats:
                cluster_stats[cluster] = {"total": 0, "no_retrieval": 0}
            cluster_stats[cluster]["total"] += 1
            if d["no_retrieval"]:
                cluster_stats[cluster]["no_retrieval"] += 1

        by_cluster = {}
        for cluster, stats in cluster_stats.items():
            by_cluster[cluster] = {
                "rate": stats["no_retrieval"] / stats["total"],
                "total": stats["total"],
                "no_retrieval": stats["no_retrieval"]
            }

        return {
            "overall_rate": overall_rate,
            "total_samples": total_samples,
            "no_retrieval_count": no_retrieval_count,
            "by_cluster": by_cluster
        }
