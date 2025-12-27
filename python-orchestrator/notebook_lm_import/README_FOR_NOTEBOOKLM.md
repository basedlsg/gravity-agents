# NotebookLM Data Import Instructions

This folder contains the "Gravity Agents" project data converted into formats optimized for Google NotebookLM (RAG).

## How to Use
1.  Open [Google NotebookLM](https://notebooklm.google.com/).
2.  Create a New Notebook.
3.  **Drag and Drop** all files in this folder (MD reports + PNG images) into the "Sources" upload area.
    *   Note: If PNGs are not accepted, just use the Markdown text files.
4.  You can now ask questions!

## Included Files

### 1. Scientific Reports (Markdown / Text)
*   `project_retrospective_qa.md`: **High Value**. The exhaustive Q&A answering 106 specific questions.
*   `Report_StressTest_N100_Parallel.md`: Data table showing the N=100 infrastructure failure.
*   `Report_Algorithm_Performance.md`: Data table showing the Agent Logic success (N=50 sample).
*   `Report_Physics_Calibration.md`: System ID data (Jump Impulse vs Distance).

### 2. High-Level Summaries
*   `final_project_summary.md`: The executive summary of the research.

### 3. Visualizations (PDF/PNG)
*   `Fig1_Success_Comparison.png`: Success Rate Bar Chart.
*   `Fig2_SystemID_Response.png`: Calibration Curve.

## Suggested Questions for NotebookLM
*   "Summarize the failure taxonomy of the agent."
*   "Why did the N=100 sweep fail compared to the N=30 baseline?"
*   "What is the relationship between Jump Impulse and Landed Distance based on the calibration data?"
*   "Draft an abstract for a paper titled 'Gravity Agents: Robust Control in Uncertain Physics'."
