#!/usr/bin/env python3
"""
System Setup Checker - Verifica configurazione INTELLIGENCE_ITA

Controlla che tutti i componenti siano configurati correttamente:
- Database PostgreSQL + pgvector
- API Keys (Gemini)
- Modelli NLP
- Dipendenze Python
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ANSI colors
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color


def print_status(check_name: str, passed: bool, message: str = ""):
    """Print check status with colors."""
    status = f"{GREEN}✓{NC}" if passed else f"{RED}✗{NC}"
    detail = f" - {message}" if message else ""
    print(f"{status} {check_name}{detail}")


def check_python_version():
    """Check Python version >= 3.9"""
    import sys
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 9
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print_status("Python Version", passed, f"v{version_str} {'✓ OK' if passed else '✗ Richiesto >= 3.9'}")
    return passed


def check_environment_file():
    """Check .env file exists and has required keys."""
    env_file = Path('.env')
    
    if not env_file.exists():
        print_status(".env File", False, "File non trovato - copia da .env.example")
        return False
    
    # Read .env
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check required keys
    has_db_url = bool(os.getenv('DATABASE_URL'))
    has_gemini = bool(os.getenv('GEMINI_API_KEY'))
    
    if has_db_url and has_gemini:
        print_status(".env File", True, "DATABASE_URL e GEMINI_API_KEY configurati")
        return True
    else:
        missing = []
        if not has_db_url:
            missing.append("DATABASE_URL")
        if not has_gemini:
            missing.append("GEMINI_API_KEY")
        print_status(".env File", False, f"Mancanti: {', '.join(missing)}")
        return False


def check_database_connection():
    """Check PostgreSQL connection."""
    try:
        from src.storage.database import DatabaseManager
        
        db = DatabaseManager()
        # Try simple connection
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
        
        db.close()
        print_status("Database Connection", True, "PostgreSQL connesso")
        return True
        
    except Exception as e:
        print_status("Database Connection", False, str(e)[:60])
        return False


def check_pgvector_extension():
    """Check pgvector extension is installed."""
    try:
        from src.storage.database import DatabaseManager
        
        db = DatabaseManager()
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
                result = cur.fetchone()
        
        db.close()
        
        if result:
            version = result[0]
            print_status("pgvector Extension", True, f"v{version} installata")
            return True
        else:
            print_status("pgvector Extension", False, "Extension non trovata")
            return False
            
    except Exception as e:
        print_status("pgvector Extension", False, str(e)[:60])
        return False


def check_database_tables():
    """Check if required tables exist."""
    try:
        from src.storage.database import DatabaseManager
        
        db = DatabaseManager()
        required_tables = ['articles', 'chunks', 'reports', 'report_feedback']
        
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
                """)
                existing_tables = [row[0] for row in cur.fetchall()]
        
        db.close()
        
        missing = [t for t in required_tables if t not in existing_tables]
        
        if not missing:
            print_status("Database Tables", True, f"{len(required_tables)} tabelle presenti")
            return True
        else:
            print_status("Database Tables", False, f"Mancanti: {', '.join(missing)}")
            print(f"  {YELLOW}→ Esegui: python scripts/load_to_database.py --init-only{NC}")
            return False
            
    except Exception as e:
        print_status("Database Tables", False, str(e)[:60])
        return False


def check_spacy_models():
    """Check spaCy models are installed."""
    try:
        import spacy
        
        # Check English model (used for articles)
        try:
            nlp_en = spacy.load("en_core_web_sm")
            print_status("spaCy Model (EN)", True, "en_core_web_sm caricato")
            has_en = True
        except:
            print_status("spaCy Model (EN)", False, "en_core_web_sm non trovato")
            print(f"  {YELLOW}→ Esegui: python -m spacy download en_core_web_sm{NC}")
            has_en = False
        
        return has_en
        
    except ImportError:
        print_status("spaCy Library", False, "spaCy non installato")
        return False


def check_sentence_transformers():
    """Check Sentence Transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
        
        # This will download if not cached (~400MB)
        # We just check it's importable
        print_status("Sentence Transformers", True, "Library disponibile")
        print(f"  {BLUE}ℹ️  Modello verrà scaricato (~400MB) al primo uso{NC}")
        return True
        
    except ImportError:
        print_status("Sentence Transformers", False, "Library non trovata")
        return False


def check_gemini_api():
    """Check Gemini API key is valid."""
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print_status("Gemini API Key", False, "GEMINI_API_KEY non trovata in .env")
            return False
        
        # Try to configure (doesn't make API call)
        genai.configure(api_key=api_key)
        
        print_status("Gemini API Key", True, "Configurata (validazione al primo uso)")
        return True
        
    except Exception as e:
        print_status("Gemini API Key", False, str(e)[:60])
        return False


def check_data_directories():
    """Check data directories exist."""
    dirs = ['data', 'reports', 'logs']
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    print_status("Data Directories", True, "data/, reports/, logs/ presenti")
    return True


def check_database_content():
    """Check if database has articles."""
    try:
        from src.storage.database import DatabaseManager
        
        db = DatabaseManager()
        stats = db.get_statistics()
        db.close()
        
        article_count = stats.get('total_articles', 0)
        chunk_count = stats.get('total_chunks', 0)
        
        if article_count > 0:
            print_status("Database Content", True, 
                        f"{article_count} articoli, {chunk_count} chunks")
            return True
        else:
            print_status("Database Content", False, "Database vuoto")
            print(f"  {YELLOW}→ Esegui pipeline completa (vedi docs/QUICKSTART.md){NC}")
            return False
            
    except Exception as e:
        print_status("Database Content", False, str(e)[:60])
        return False


def main():
    """Run all checks."""
    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}INTELLIGENCE_ITA - System Setup Check{NC}")
    print(f"{BLUE}{'='*60}{NC}\n")
    
    checks = [
        ("Python Environment", check_python_version),
        ("Environment File", check_environment_file),
        ("Database Connection", check_database_connection),
        ("pgvector Extension", check_pgvector_extension),
        ("Database Tables", check_database_tables),
        ("Database Content", check_database_content),
        ("spaCy Models", check_spacy_models),
        ("Sentence Transformers", check_sentence_transformers),
        ("Gemini API", check_gemini_api),
        ("Data Directories", check_data_directories),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print_status(check_name, False, f"Exception: {str(e)[:50]}")
            results.append((check_name, False))
    
    # Summary
    print(f"\n{BLUE}{'='*60}{NC}")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    if passed == total:
        print(f"{GREEN}✓ Tutti i controlli superati ({passed}/{total}){NC}")
        print(f"\n{GREEN}Il sistema è pronto per l'uso!{NC}")
        print(f"\nProssimi passi:")
        print(f"  1. Esegui pipeline: python -m src.ingestion.pipeline")
        print(f"  2. Process NLP: python scripts/process_nlp.py")
        print(f"  3. Load DB: python scripts/load_to_database.py")
        print(f"  4. Generate report: python scripts/generate_report.py")
        print(f"  5. Open dashboard: ./scripts/run_dashboard.sh")
        return 0
    else:
        failed = total - passed
        print(f"{YELLOW}⚠ {failed} controlli falliti su {total}{NC}")
        print(f"\nRisolvi i problemi sopra indicati e riesegui questo script.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Interrupted by user{NC}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{RED}Unexpected error: {e}{NC}")
        sys.exit(1)
