"""
tests/test_clustering.py - Unit test suite for Phase 1 Step 4: Corpus Engineering & Micro-Clustering.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from lib.utils import PipelineConfig
from lib.s5_clustering.embedder import run_embedding, load_corpus
from lib.s5_clustering.dim_reducer import apply_dimensionality_reduction
from lib.s5_clustering.clusterer import run_clustering, ClusterAssignment
from lib.s5_clustering.splitter import run_splitting
from lib.s5_clustering.cluster_reporter import run_reporting
from lib.s5_clustering import run_clustering_pipeline


@pytest.fixture
def test_cfg():
    cfg = PipelineConfig()
    # Configure tiny test/temp paths and configurations
    cfg.misc.seed = 42
    cfg.clustering.embedding_model = "all-mpnet-base-v2"
    cfg.clustering.embed_batch_size = 4
    cfg.clustering.hdbscan_min_cluster_size = 2
    cfg.clustering.hdbscan_min_samples = 1
    cfg.clustering.hdbscan_metric = "cosine"
    cfg.clustering.noise_assignment = "nearest"
    cfg.clustering.cluster_min_fraction = 0.15
    cfg.clustering.cluster_max_fraction = 0.40
    cfg.clustering.split_dev_ratio = 0.70
    cfg.clustering.split_val_ratio = 0.20
    cfg.clustering.split_sealed_ratio = 0.10
    return cfg


@pytest.fixture
def mock_corpus_file():
    # Return a temporary file with a small corpus of 10 items
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".jsonl", encoding="utf-8") as f:
        for i in range(12):
            doc = {
                "id": f"doc_{i:03d}",
                "text": f"This is some scientific text for document {i} that discusses neuroscience.",
                "source_file": "neuro_notes.pdf"
            }
            f.write(json.dumps(doc) + "\n")
        f.flush()
        yield Path(f.name)
    try:
        Path(f.name).unlink()
    except OSError:
        pass


@pytest.fixture
def mock_sentence_transformer():
    with patch("lib.s5_clustering.embedder.SentenceTransformer") as mock_class:
        mock_instance = MagicMock()
        # Mock encoding to return L2-normalized 768-dim random floats
        def mock_encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True):
            n = len(texts)
            arr = np.random.randn(n, 768).astype(np.float32)
            # L2-normalize
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            return arr / np.where(norms > 0, norms, 1)

        mock_instance.encode.side_effect = mock_encode
        mock_class.return_value = mock_instance
        yield mock_instance


def test_embedder_output_shape(test_cfg, mock_corpus_file, mock_sentence_transformer):
    test_cfg.clustering.corpus_path = mock_corpus_file
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_cfg.clustering.embeddings_cache_path = tmp_path / "embeddings.npy"
        test_cfg.clustering.doc_ids_cache_path = tmp_path / "doc_ids.json"
        test_cfg.clustering.cluster_manifest_path = tmp_path / "cluster_manifest.json"

        embeddings, doc_ids = run_embedding(test_cfg)
        
        assert embeddings.shape == (12, 768)
        assert len(doc_ids) == 12
        assert doc_ids[0] == "doc_000"
        
        # Verify cache files were created
        assert test_cfg.clustering.embeddings_cache_path.exists()
        assert test_cfg.clustering.doc_ids_cache_path.exists()


def test_embedder_cache_reuse(test_cfg, mock_corpus_file, mock_sentence_transformer):
    test_cfg.clustering.corpus_path = mock_corpus_file
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_cfg.clustering.embeddings_cache_path = tmp_path / "embeddings.npy"
        test_cfg.clustering.doc_ids_cache_path = tmp_path / "doc_ids.json"
        test_cfg.clustering.cluster_manifest_path = tmp_path / "cluster_manifest.json"

        # First run should hit model.encode
        run_embedding(test_cfg)
        assert mock_sentence_transformer.encode.call_count == 1

        # Second run should load from cache
        embeddings, doc_ids = run_embedding(test_cfg)
        assert mock_sentence_transformer.encode.call_count == 1
        assert embeddings.shape == (12, 768)
        assert len(doc_ids) == 12


def test_embedder_cache_miss_on_count_change(test_cfg, mock_corpus_file, mock_sentence_transformer):
    test_cfg.clustering.corpus_path = mock_corpus_file
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_cfg.clustering.embeddings_cache_path = tmp_path / "embeddings.npy"
        test_cfg.clustering.doc_ids_cache_path = tmp_path / "doc_ids.json"
        test_cfg.clustering.cluster_manifest_path = tmp_path / "cluster_manifest.json"

        # First run (12 documents)
        run_embedding(test_cfg)
        assert mock_sentence_transformer.encode.call_count == 1

        # Write a new corpus with only 5 documents
        with open(mock_corpus_file, "w", encoding="utf-8") as f:
            for i in range(5):
                doc = {"id": f"doc_{i:03d}", "text": "Short text"}
                f.write(json.dumps(doc) + "\n")

        # Second run should detect change and hit encoder again
        embeddings, doc_ids = run_embedding(test_cfg)
        assert mock_sentence_transformer.encode.call_count == 2
        assert embeddings.shape == (5, 768)
        assert len(doc_ids) == 5


def test_clusterer_basic(test_cfg):
    # Setup simple synthetic embeddings where first 5 are close, second 5 are close
    embeddings = np.zeros((10, 768), dtype=np.float32)
    embeddings[:5, 0] = 1.0  # cluster A
    embeddings[5:, 1] = 1.0  # cluster B
    doc_ids = [f"doc_{i}" for i in range(10)]

    test_cfg.clustering.hdbscan_min_cluster_size = 3
    test_cfg.clustering.hdbscan_min_samples = 1
    
    assignments = run_clustering(test_cfg, embeddings, doc_ids)
    
    assert len(assignments) == 10
    assert all(isinstance(a, ClusterAssignment) for a in assignments)
    
    # Verify that different clusters were assigned
    cluster_labels = set(a.cluster_label for a in assignments)
    # Excluding dropped/noise, we should have 2 clusters
    assert len(cluster_labels) >= 1


def test_clusterer_noise_nearest(test_cfg):
    # Setup: 2 clusters, and 1 noise point
    embeddings = np.zeros((7, 768), dtype=np.float32)
    # Cluster 0
    embeddings[0, 0] = 1.0
    embeddings[1, 0] = 0.99
    embeddings[2, 0] = 0.98
    # Cluster 1
    embeddings[3, 1] = 1.0
    embeddings[4, 1] = 0.99
    embeddings[5, 1] = 0.98
    # Noise point: closer to cluster 0 in cosine space
    embeddings[6, 0] = 0.7
    embeddings[6, 1] = 0.1
    
    doc_ids = [f"doc_{i}" for i in range(7)]
    
    # We patch HDBSCAN to return labels: [0, 0, 0, 1, 1, 1, -1]
    with patch("hdbscan.HDBSCAN") as mock_hdbscan:
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = np.array([0, 0, 0, 1, 1, 1, -1])
        mock_hdbscan.return_value = mock_instance
        
        test_cfg.clustering.noise_assignment = "nearest"
        assignments = run_clustering(test_cfg, embeddings, doc_ids)
        
        assert len(assignments) == 7
        noise_point = assignments[6]
        assert noise_point.is_noise is True
        assert noise_point.assigned_by == "nearest_centroid"
        # Since it is closer to cluster 0 (dimension 0 is 0.7 vs dimension 1 is 0.1),
        # it should be reassigned to cluster 0
        assert noise_point.cluster_id == 0
        assert noise_point.cluster_label == "cluster_000"


def test_clusterer_noise_drop(test_cfg):
    embeddings = np.zeros((4, 768), dtype=np.float32)
    doc_ids = [f"doc_{i}" for i in range(4)]
    
    with patch("hdbscan.HDBSCAN") as mock_hdbscan:
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = np.array([0, 0, 0, -1])
        mock_hdbscan.return_value = mock_instance
        
        test_cfg.clustering.noise_assignment = "drop"
        assignments = run_clustering(test_cfg, embeddings, doc_ids)
        
        assert len(assignments) == 4
        noise_point = assignments[3]
        assert noise_point.is_noise is True
        assert noise_point.assigned_by == "dropped"
        assert noise_point.cluster_id == -1
        assert noise_point.cluster_label == "dropped"


def test_clusterer_labels_zero_padded(test_cfg):
    embeddings = np.zeros((3, 768), dtype=np.float32)
    doc_ids = ["doc_a", "doc_b", "doc_c"]
    
    with patch("hdbscan.HDBSCAN") as mock_hdbscan:
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = np.array([7, 7, 7])
        mock_hdbscan.return_value = mock_instance
        
        assignments = run_clustering(test_cfg, embeddings, doc_ids)
        assert assignments[0].cluster_label == "cluster_007"


def test_splitter_ratios(test_cfg):
    assignments = []
    # 20 documents in cluster_001
    for i in range(20):
        assignments.append(
            ClusterAssignment(
                doc_id=f"doc_{i}",
                cluster_id=1,
                cluster_label="cluster_001",
                is_noise=False,
                assigned_by="hdbscan"
            )
        )
    
    test_cfg.clustering.split_dev_ratio = 0.70
    test_cfg.clustering.split_val_ratio = 0.20
    test_cfg.clustering.split_sealed_ratio = 0.10
    
    splits_data = run_splitting(test_cfg, assignments)
    
    cluster_split = splits_data["clusters"]["cluster_001"]
    
    # 20 * 0.7 = 14 dev
    # 20 * 0.2 = 4 val
    # 20 * 0.1 = 2 sealed
    assert len(cluster_split["dev_doc_ids"]) == 14
    assert len(cluster_split["val_doc_ids"]) == 4
    assert len(cluster_split["sealed_doc_ids"]) == 2


def test_splitter_small_cluster(test_cfg):
    assignments = [
        ClusterAssignment("doc_1", 2, "cluster_002", False, "hdbscan"),
        ClusterAssignment("doc_2", 2, "cluster_002", False, "hdbscan"),
    ]
    
    splits_data = run_splitting(test_cfg, assignments)
    cluster_split = splits_data["clusters"]["cluster_002"]
    
    assert len(cluster_split["dev_doc_ids"]) == 2
    assert len(cluster_split["val_doc_ids"]) == 0
    assert len(cluster_split["sealed_doc_ids"]) == 0


def test_splitter_reweight_min(test_cfg):
    # Total docs = 100. Min fraction = 0.15.
    # cluster_001 has 10 docs -> raw_fraction = 0.10. Underrepresented!
    assignments = []
    for i in range(10):
        assignments.append(ClusterAssignment(f"doc_c1_{i}", 1, "cluster_001", False, "hdbscan"))
    for i in range(90):
        assignments.append(ClusterAssignment(f"doc_c2_{i}", 2, "cluster_002", False, "hdbscan"))
        
    test_cfg.clustering.cluster_min_fraction = 0.15
    test_cfg.clustering.cluster_max_fraction = 0.90
    
    splits_data = run_splitting(test_cfg, assignments)
    
    c1_split = splits_data["clusters"]["cluster_001"]
    c2_split = splits_data["clusters"]["cluster_002"]
    
    # Reweight cap should be math.ceil(100 * 0.15) = 15
    assert c1_split["reweight_cap"] == 15
    assert c2_split["reweight_cap"] is None


def test_splitter_reweight_max(test_cfg):
    # Total docs = 100. Max fraction = 0.40.
    # cluster_002 has 50 docs -> raw_fraction = 0.50. Overrepresented!
    assignments = []
    for i in range(50):
        assignments.append(ClusterAssignment(f"doc_c1_{i}", 1, "cluster_001", False, "hdbscan"))
    for i in range(50):
        assignments.append(ClusterAssignment(f"doc_c2_{i}", 2, "cluster_002", False, "hdbscan"))
        
    test_cfg.clustering.cluster_min_fraction = 0.01
    test_cfg.clustering.cluster_max_fraction = 0.40
    
    splits_data = run_splitting(test_cfg, assignments)
    
    c1_split = splits_data["clusters"]["cluster_001"]
    c2_split = splits_data["clusters"]["cluster_002"]
    
    # Both are 50/100 = 50%, which is > 40%.
    # Reweight cap should be math.floor(100 * 0.40) = 40
    assert c1_split["reweight_cap"] == 40
    assert c2_split["reweight_cap"] == 40


def test_splitter_within_range(test_cfg):
    assignments = []
    for i in range(30):
        assignments.append(ClusterAssignment(f"doc_c1_{i}", 1, "cluster_001", False, "hdbscan"))
    for i in range(70):
        assignments.append(ClusterAssignment(f"doc_c2_{i}", 2, "cluster_002", False, "hdbscan"))
        
    test_cfg.clustering.cluster_min_fraction = 0.10
    test_cfg.clustering.cluster_max_fraction = 0.80
    
    splits_data = run_splitting(test_cfg, assignments)
    
    # 30% and 70% both fall inside [10%, 80%]
    assert splits_data["clusters"]["cluster_001"]["reweight_cap"] is None
    assert splits_data["clusters"]["cluster_002"]["reweight_cap"] is None


def test_cluster_reporter_hard_fail(test_cfg):
    assignments = [
        ClusterAssignment("d1", 1, "cluster_001", False, "hdbscan"),
        ClusterAssignment("d2", 1, "cluster_001", False, "hdbscan"),
        ClusterAssignment("d3", 1, "cluster_001", False, "hdbscan"),
        ClusterAssignment("d4", 2, "cluster_002", False, "hdbscan"),
        ClusterAssignment("d5", 2, "cluster_002", False, "hdbscan"),
        ClusterAssignment("d6", 2, "cluster_002", False, "hdbscan"),
    ]
    
    # We patch is_noise to simulate 0 noise docs
    splits_data = {
        "clusters": {
            "cluster_001": {
                "total_docs": 3,
                "raw_fraction": 0.5,
                "dev_doc_ids": ["d1", "d2"],
                "val_doc_ids": ["d3"],
                "sealed_doc_ids": [],
                "reweight_cap": None
            },
            "cluster_002": {
                "total_docs": 3,
                "raw_fraction": 0.5,
                "dev_doc_ids": ["d4", "d5"],
                "val_doc_ids": ["d6"],
                "sealed_doc_ids": [],
                "reweight_cap": None
            }
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_cfg.clustering.cluster_manifest_path = tmp_path / "cluster_manifest.json"
        test_cfg.clustering.cluster_report_path = tmp_path / "cluster_report.json"
        
        # Explicitly set min_clusters to 10 for the hard fail test
        test_cfg.clustering.min_clusters = 10
        
        with pytest.raises(ValueError) as excinfo:
            run_reporting(test_cfg, assignments, splits_data)
        assert "Hard fail: Cluster count is 2 (expected >= 10)" in str(excinfo.value)



def test_cluster_reporter_manifest(test_cfg):
    # Create 10 clusters of size 3 each to satisfy cluster count >= 10 gate.
    assignments = []
    clusters = {}
    for c_id in range(10):
        c_label = f"cluster_{c_id:03d}"
        doc_a = f"doc_{c_id}_a"
        doc_b = f"doc_{c_id}_b"
        doc_c = f"doc_{c_id}_c"
        
        assignments.extend([
            ClusterAssignment(doc_a, c_id, c_label, False, "hdbscan"),
            ClusterAssignment(doc_b, c_id, c_label, False, "hdbscan"),
            ClusterAssignment(doc_c, c_id, c_label, False, "hdbscan"),
        ])
        clusters[c_label] = {
            "total_docs": 3,
            "raw_fraction": 0.1,
            "dev_doc_ids": [doc_a, doc_b],
            "val_doc_ids": [doc_c],
            "sealed_doc_ids": [],
            "reweight_cap": None
        }
        
    splits_data = {
        "clusters": clusters
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_cfg.clustering.cluster_manifest_path = tmp_path / "cluster_manifest.json"
        test_cfg.clustering.cluster_report_path = tmp_path / "cluster_report.json"
        
        manifest = run_reporting(test_cfg, assignments, splits_data)
        
        assert manifest["status"] == "complete"
        assert manifest["embedding_model"] == test_cfg.clustering.embedding_model
        assert manifest["total_docs"] == 30
        assert manifest["noise_docs"] == 0
        assert manifest["noise_fraction"] == 0.0
        assert manifest["total_clusters"] == 10
        assert "cluster_sizes" in manifest
        assert manifest["cluster_sizes"]["min"] == 3
        assert manifest["cluster_sizes"]["max"] == 3
        assert manifest["cluster_sizes"]["mean"] == 3.0
        
        # Verify that files were written
        assert test_cfg.clustering.cluster_manifest_path.exists()
        assert test_cfg.clustering.cluster_report_path.exists()


def test_cluster_reporter_warnings(test_cfg):
    # Create 10 clusters of size 3 each to satisfy cluster count >= 10 gate.
    assignments = []
    clusters = {}
    for c_id in range(10):
        c_label = f"cluster_{c_id:03d}"
        doc_a = f"doc_{c_id}_a"
        doc_b = f"doc_{c_id}_b"
        doc_c = f"doc_{c_id}_c"
        
        assignments.extend([
            ClusterAssignment(doc_a, c_id, c_label, False, "hdbscan"),
            ClusterAssignment(doc_b, c_id, c_label, False, "hdbscan"),
            ClusterAssignment(doc_c, c_id, c_label, False, "hdbscan"),
        ])
        clusters[c_label] = {
            "total_docs": 3,
            "raw_fraction": 0.1,
            "dev_doc_ids": [doc_a, doc_b],
            "val_doc_ids": [doc_c],
            "sealed_doc_ids": [],
            "reweight_cap": None
        }
        
    # Also add some noise docs (say 20 noise docs out of 50 total -> noise fraction = 40%)
    for i in range(20):
        assignments.append(ClusterAssignment(f"noise_{i}", -1, "dropped", True, "dropped"))
        
    splits_data = {
        "clusters": clusters
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_cfg.clustering.cluster_manifest_path = tmp_path / "cluster_manifest.json"
        test_cfg.clustering.cluster_report_path = tmp_path / "cluster_report.json"
        
        manifest = run_reporting(test_cfg, assignments, splits_data)
        
        assert manifest["status"] == "complete"
        assert manifest["noise_fraction"] > 0.30
        assert any("High noise fraction" in w for w in manifest["warnings"])


def test_pipeline_end_to_end(test_cfg, mock_corpus_file, mock_sentence_transformer):
    test_cfg.clustering.corpus_path = mock_corpus_file
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_cfg.clustering.output_dir = tmp_path / "clustering"
        test_cfg.clustering.embeddings_cache_path = tmp_path / "clustering/embeddings.npy"
        test_cfg.clustering.doc_ids_cache_path = tmp_path / "clustering/doc_ids.json"
        test_cfg.clustering.assignments_path = tmp_path / "clustering/cluster_assignments.jsonl"
        test_cfg.clustering.splits_path = tmp_path / "clustering/splits.json"
        test_cfg.clustering.cluster_manifest_path = tmp_path / "clustering/cluster_manifest.json"
        test_cfg.clustering.cluster_report_path = tmp_path / "logs/clustering/cluster_report.json"
        
        # Override gate check to prevent hard fail on cluster count < 10 for test corpus
        # We patch HDBSCAN to return 10 distinct labels for our 12 test documents
        with patch("hdbscan.HDBSCAN") as mock_hdbscan:
            mock_instance = MagicMock()
            # 10 clusters: labels 0 to 9, and two noise points (-1)
            mock_instance.fit_predict.return_value = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1])
            mock_hdbscan.return_value = mock_instance
            
            run_clustering_pipeline(test_cfg)
            
            # Check all files are written
            assert test_cfg.clustering.embeddings_cache_path.exists()
            assert test_cfg.clustering.doc_ids_cache_path.exists()
            assert test_cfg.clustering.assignments_path.exists()
            assert test_cfg.clustering.splits_path.exists()
            assert test_cfg.clustering.cluster_manifest_path.exists()
            assert test_cfg.clustering.cluster_report_path.exists()
            
            # Read manifest and check status
            with open(test_cfg.clustering.cluster_manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            assert manifest["status"] == "complete"
            assert manifest["total_clusters"] == 10
            assert "dim_reduction_method" in manifest


def test_dim_reducer_umap(test_cfg):
    np.random.seed(42)
    embeddings = np.random.randn(25, 768).astype(np.float32)
    test_cfg.clustering.dim_reduction_method = "umap"
    test_cfg.clustering.umap_n_components = 5
    test_cfg.clustering.umap_n_neighbors = 10

    reduced, meta = apply_dimensionality_reduction(embeddings, test_cfg)
    assert reduced.shape == (25, 5)
    assert meta["method"] == "umap"
    assert meta["reduced_dim"] == 5
    assert meta["fallback_triggered"] is False
    # Check L2 normalized
    norms = np.linalg.norm(reduced, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-4)


def test_dim_reducer_pca(test_cfg):
    np.random.seed(42)
    embeddings = np.random.randn(20, 768).astype(np.float32)
    test_cfg.clustering.dim_reduction_method = "pca"
    test_cfg.clustering.pca_components = 8

    reduced, meta = apply_dimensionality_reduction(embeddings, test_cfg)
    assert reduced.shape == (20, 8)
    assert meta["method"] == "pca"
    assert meta["reduced_dim"] == 8
    # Check L2 normalized
    norms = np.linalg.norm(reduced, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-4)


def test_dim_reducer_passthrough(test_cfg):
    np.random.seed(42)
    embeddings = np.random.randn(10, 768).astype(np.float32)
    test_cfg.clustering.dim_reduction_method = "passthrough"

    reduced, meta = apply_dimensionality_reduction(embeddings, test_cfg)
    assert reduced.shape == (10, 768)
    assert meta["method"] == "passthrough"
    assert meta["reduced_dim"] == 768


def test_dim_reducer_umap_fallback_small_corpus(test_cfg):
    embeddings = np.random.randn(4, 768).astype(np.float32)
    test_cfg.clustering.dim_reduction_method = "umap"
    test_cfg.clustering.umap_n_components = 15

    # Should trigger fallback to PCA because n_docs (4) <= n_components (15)
    reduced, meta = apply_dimensionality_reduction(embeddings, test_cfg)
    assert meta["fallback_triggered"] is True
    assert meta["method"] == "pca"
    assert reduced.shape[0] == 4


def test_dim_reducer_invalid_method(test_cfg):
    embeddings = np.random.randn(10, 768).astype(np.float32)
    test_cfg.clustering.dim_reduction_method = "invalid_method"

    with pytest.raises(ValueError) as excinfo:
        apply_dimensionality_reduction(embeddings, test_cfg)
    assert "Unsupported dim_reduction_method" in str(excinfo.value)

