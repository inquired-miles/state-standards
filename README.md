# State Standards Analysis System

A Django application that uses PostgreSQL with pgvector to store and analyze state educational standards data. The system allows for finding correlations between standards across different states using vector embeddings and similarity search. Redis is used for caching and session storage.

## Features

- Store state educational standards with vector embeddings
- Find correlations between standards across different states
- Django admin interface for data management
- Vector similarity search using pgvector
- Supabase integration for scalable storage
- Batch processing commands for data operations

## Setup

### Prerequisites

- Python 3.11+
- Anaconda/Miniconda
- PoistgreSQL 14+
- Redis Server
- OpenAI API key (for generating embeddings)

### Installation

1. **Create and activate conda environment:**
   ```bash
   conda create -n state-standards python=3.11 -y
   conda activate state-standards
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Copy `.env.example` to `.env` and fill in your actual values:
   ```bash
   cp .env.example .env
   ```
   
   Update the following variables in your `.env` file:
   - `DATABASE_URL`: Your PostgreSQL connection string with pgvector enabled
   - `OPENAI_API_KEY`: Your OpenAI API key for generating embeddings
   - `SECRET_KEY`: Django secret key
    - `REDIS_URL`: Redis connection string (e.g. `redis://127.0.0.1:6379/0`)

4. **Run migrations:**
   ```bash
   python manage.py migrate
   ```

5. **Create superuser:**
   ```bash
   python manage.py createsuperuser
   ```

## Usage

### Loading Sample Data

Load sample state standards data:
```bash
python manage.py load_sample_data
```

### Generating Embeddings

Generate vector embeddings for all standards:
```bash
python manage.py generate_embeddings
```

### Creating Correlations

Create correlations between standards based on similarity:
```bash
python manage.py create_correlations --threshold 0.8
```

Create correlations for a specific standard:
```bash
python manage.py create_correlations --standard-code "CA.1.OA.1" --threshold 0.8
```

### Admin Interface

Access the Django admin interface at `http://localhost:8000/admin/` to:
- Manage states, subject areas, and grade levels
- Add and edit standards
- View and verify correlations between standards
- Perform bulk operations

### Running the Development Server

```bash
python manage.py runserver
```

## Database Schema

### Models

- **State**: Represents US states (code, name)
- **SubjectArea**: Subject areas like Mathematics, ELA, etc.
- **GradeLevel**: Grade levels from K-12
- **Standard**: Educational standards with vector embeddings
- **StandardCorrelation**: Correlations between standards with similarity scores

### Vector Operations

The system uses pgvector for efficient similarity search:
- Standards are embedded using OpenAI's text-embedding-3-small model
- Cosine similarity is used to find related standards
- Correlations are automatically created based on similarity thresholds

## API Integration

### OpenAI

Uses OpenAI's Embedding API to:
- Generate vector embeddings for standard text
- Enable semantic similarity search
- Support cross-state standard correlation analysis

## Development

### Running Tests

```bash
python manage.py test
```

### Linting and Code Quality

```bash
# Install development dependencies
pip install flake8 black isort

# Format code
black .
isort .

# Check code quality
flake8 .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request
