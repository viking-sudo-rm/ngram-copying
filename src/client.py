import requests

class RustyDawgClient:
    """Make single request to an API."""

    def __init__(self, host: str = "localhost:5000"):
        self.host = host

    def query(self, text):
        url = f"http://{self.host}/api/cdawg"
        res = requests.post(url, json={"text": text})
        return res.json()

def test_normal():
    text = ["Four score and seven years ago, Rusty DAWG was launched."]
    client = RustyDawgClient()
    print("Got:", client.query(text))

if __name__ == "__main__":
    test_normal()    
