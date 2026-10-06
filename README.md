Requires 
- macOS on Apple Silicon
- Python 3.14 or higher

```
python3 -m venv .venv

source .venv/bin/activate

pip3 install -r requirements.txt

docker run -d -p 6333:6333 qdrant/qdrant

python3 main.py
```

Wait for the > prompt.

/q to quit.
