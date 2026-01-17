# CiteFlow

CiteFlow is a powerful reference verification tool designed to validate academic references using multiple sources. It leverages the Crossref API, ArXiv API, and web verification to ensure the accuracy of citations.

## Features

- **Multi-Source Verification**:
  - **ArXiv**: Automatically detects and verifies ArXiv IDs.
  - **Crossref**: Fuzzy matches reference strings against the Crossref database.
  - **Web Verification**: Scrapes DOI landing pages as a fallback to verify titles and authors when fuzzy scores are inconclusive.
- **Automated Monitoring**: Watch a directory for new reference lists and process them automatically.
- **Intelligent Ranking**: Uses `rapidfuzz` to score and rank potential matches.
- **Detailed Reporting**: Classifies results into `Verified` and `Manual_Check` categories for easy review.

## Setup

### 1. Prerequisites
- Python 3.10+
- An active internet connection.

### 2. Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configuration

CiteFlow requires a Crossref-registered email to be set in a `.env` file for polite API usage.

Create a `.env` file in the root directory:

```env
CROSSREF_EMAIL=your_email@example.com
```

## Usage

### Standard Verification

Run the script manually on a text file containing references (one per line):

```bash
python src/cite_flow_core.py path/to/references.txt
```

If no file is provided, it defaults to checking `data/samples/2312.01797v3_references.txt`.

### Monitor Mode

To automatically process new files, use the monitoring mode. This watches the `data/samples` directory for new `.txt` files.

```bash
python src/cite_flow_core.py --monitor
```

1. Start the monitor.
2. Drop a text file (e.g., `my_paper_refs.txt`) into `data/samples`.
3. The script detects the file and processes it.
4. Results are saved in `Results/my_paper_refs/` (organized into `Verified` and `Manual_Check` folders).
5. Press `Ctrl+C` to stop monitoring.

## Output

Results are JSON files containing:
- Matched Title & Authors
- Similarity Score
- DOI / ArXiv ID
- Verification Method Used

Directory structure of results:
```
Results/
└── <input_filename>/
    ├── Verified/       # High confidence matches
    └── Manual_Check/   # Low confidence or ambiguous matches
```

## Architecture

The system is built with a modular architecture:
- **Pipeline**: Manages the flow of data.
- **Services**: `ArxivService`, `CrossrefService`, `WebVerifier`.
- **Engine**: `VerificationEngine` encapsulates the decision logic.
- **Watcher**: `ReferenceHandler` (Watchdog) for file monitoring.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
