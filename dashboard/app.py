"""Streamlit dashboard skeleton for benchmark result exploration."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


BENCHMARK_ROWS = [
    {
        "Dataset": "Small blobs",
        "Samples": 1_000,
        "Features": 16,
        "Backend": "CPU",
        "Runtime (s)": 0.18,
        "Memory (MB)": 42,
        "Status": "Baseline",
    },
    {
        "Dataset": "Small blobs",
        "Samples": 1_000,
        "Features": 16,
        "Backend": "GPU",
        "Runtime (s)": 0.09,
        "Memory (MB)": 58,
        "Status": "Target",
    },
    {
        "Dataset": "Medium blobs",
        "Samples": 10_000,
        "Features": 32,
        "Backend": "CPU",
        "Runtime (s)": 2.64,
        "Memory (MB)": 210,
        "Status": "Baseline",
    },
    {
        "Dataset": "Medium blobs",
        "Samples": 10_000,
        "Features": 32,
        "Backend": "GPU",
        "Runtime (s)": 0.84,
        "Memory (MB)": 275,
        "Status": "Target",
    },
    {
        "Dataset": "Large blobs",
        "Samples": 100_000,
        "Features": 50,
        "Backend": "CPU",
        "Runtime (s)": 23.80,
        "Memory (MB)": 1_120,
        "Status": "Baseline",
    },
    {
        "Dataset": "Large blobs",
        "Samples": 100_000,
        "Features": 50,
        "Backend": "GPU",
        "Runtime (s)": 6.75,
        "Memory (MB)": 1_460,
        "Status": "Target",
    },
]


def _filter_rows(dataset: str, backend: str) -> list[dict[str, object]]:
    rows = BENCHMARK_ROWS
    if dataset != "All datasets":
        rows = [row for row in rows if row["Dataset"] == dataset]
    if backend != "All backends":
        rows = [row for row in rows if row["Backend"] == backend]
    return rows


def _build_runtime_chart(rows: list[dict[str, object]]) -> go.Figure:
    figure = go.Figure()
    for backend in sorted({str(row["Backend"]) for row in rows}):
        backend_rows = [row for row in rows if row["Backend"] == backend]
        figure.add_trace(
            go.Bar(
                name=backend,
                x=[str(row["Dataset"]) for row in backend_rows],
                y=[float(row["Runtime (s)"]) for row in backend_rows],
            )
        )
    figure.update_layout(
        title="Runtime by backend",
        xaxis_title="Dataset",
        yaxis_title="Runtime (seconds)",
        barmode="group",
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
    )
    return figure


def _build_memory_chart(rows: list[dict[str, object]]) -> go.Figure:
    figure = go.Figure(
        data=[
            go.Scatter(
                x=[str(row["Dataset"]) for row in rows],
                y=[float(row["Memory (MB)"]) for row in rows],
                mode="markers+lines",
                text=[str(row["Backend"]) for row in rows],
                marker={"size": 10},
            )
        ]
    )
    figure.update_layout(
        title="Memory footprint",
        xaxis_title="Dataset",
        yaxis_title="Memory (MB)",
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
    )
    return figure


def main() -> None:
    st.set_page_config(
        page_title="HPC Clustering Benchmarks",
        layout="wide",
    )

    st.title("HPC Clustering Benchmark Dashboard")
    st.caption("Starter dashboard with dummy CPU/GPU benchmark data.")

    dataset_options = ["All datasets"] + sorted(
        {str(row["Dataset"]) for row in BENCHMARK_ROWS}
    )
    backend_options = ["All backends"] + sorted(
        {str(row["Backend"]) for row in BENCHMARK_ROWS}
    )

    with st.sidebar:
        st.header("Filters")
        dataset = st.selectbox("Dataset", dataset_options)
        backend = st.selectbox("Backend", backend_options)
        st.info("Replace the dummy rows with benchmark output files as the engine matures.")

    rows = _filter_rows(dataset=dataset, backend=backend)

    average_runtime = sum(float(row["Runtime (s)"]) for row in rows) / len(rows)
    fastest_row = min(rows, key=lambda row: float(row["Runtime (s)"]))
    max_samples = max(int(row["Samples"]) for row in rows)

    metric_columns = st.columns(3)
    metric_columns[0].metric("Runs displayed", len(rows))
    metric_columns[1].metric("Average runtime", f"{average_runtime:.2f}s")
    metric_columns[2].metric(
        "Largest dataset",
        f"{max_samples:,} samples",
        help=f"Fastest visible run: {fastest_row['Backend']} on {fastest_row['Dataset']}",
    )

    st.subheader("Benchmark results")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    chart_columns = st.columns(2)
    chart_columns[0].plotly_chart(
        _build_runtime_chart(rows),
        use_container_width=True,
    )
    chart_columns[1].plotly_chart(
        _build_memory_chart(rows),
        use_container_width=True,
    )

    st.subheader("Next integration points")
    st.markdown(
        """
        - Load benchmark result files from `data/` or a future `benchmark_results/` directory.
        - Add CPU/GPU backend toggles once GPU benchmark output is available.
        - Track throughput, convergence iterations, and memory transfer overhead.
        """
    )


if __name__ == "__main__":
    main()
