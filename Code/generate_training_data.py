"""
Generate the labelled routing dataset used to train the PRISM soft gating module.

PRISM routes a prompt to one of three execution paths based on its privacy
sensitivity (Figure 2 / Algorithm 1 in the paper):

    common  (no sensitive entities)            -> cloud-only
    tourism / medical (moderate sensitivity)   -> collaborative
    banking (highly sensitive, e.g. card no.)  -> edge-only

The 40 medical prompts are the authentic ones shipped in
``Dataset/prism_dataset.xlsx``. The tourism / banking / common splits are
representative, regenerated examples (the original semi-synthetic splits used
for the paper are not redistributable). Routing labels follow the policy above.

The output of this script (``Dataset/routing_dataset.xlsx``) is the supervision
signal for ``train_soft_gating.py``.
"""

import os
import random
import pandas as pd

SEED = 42
N_PER_DOMAIN = 40
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "Dataset", "routing_dataset.xlsx")


def luhn_complete(prefix_digits):
    """Return a card number string whose final check digit makes it Luhn-valid."""
    digits = [int(d) for d in prefix_digits]
    # Compute the checksum over the existing digits (these become the high-order
    # positions once the check digit is appended on the right).
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:  # positions that will be doubled once check digit is added
            d *= 2
            if d > 9:
                d -= 9
        total += d
    check = (10 - (total % 10)) % 10
    return "".join(prefix_digits) + str(check)


def make_card(rng):
    prefix = ["4"] + [str(rng.randint(0, 9)) for _ in range(14)]  # 15 digits -> +1 check
    return luhn_complete(prefix)


def gen_common(rng):
    questions = [
        "What is the capital of France?",
        "How many integers satisfy the inequality |x + 5| < 10?",
        "Explain how photosynthesis works in simple terms.",
        "What is the boiling point of water at sea level?",
        "Summarize the plot of Romeo and Juliet in two sentences.",
        "How do I convert 45 degrees Fahrenheit to Celsius?",
        "What are the first ten prime numbers?",
        "Give me a simple recipe for pancakes.",
        "What is the difference between weather and climate?",
        "How does a combustion engine work?",
        "What year did the first airplane fly?",
        "Explain the Pythagorean theorem.",
        "What is the largest planet in the solar system?",
        "How many continents are there on Earth?",
        "Define supply and demand in economics.",
        "What is the speed of light in a vacuum?",
        "List three renewable energy sources.",
        "How do vaccines work in general terms?",
        "What is the chemical formula for table salt?",
        "Explain what an algorithm is.",
        "What causes the seasons to change?",
        "How do I calculate the area of a circle?",
        "What is the tallest mountain in the world?",
        "Describe the water cycle briefly.",
        "What is the difference between RAM and storage?",
        "How many sides does a hexagon have?",
        "What is the function of the heart?",
        "Explain gravity to a ten-year-old.",
        "What language has the most native speakers?",
        "How does a rainbow form?",
        "What is the square root of 144?",
        "Name three states of matter.",
        "What is the purpose of a thesis statement?",
        "How does an electric car differ from a gas car?",
        "What is the freezing point of water in Celsius?",
        "Explain what compound interest means.",
        "What is the smallest country in the world?",
        "How do bees make honey?",
        "What is the difference between a virus and bacteria?",
        "Convert 2 kilometers into miles.",
    ]
    rng.shuffle(questions)
    return [{"prompt": q, "route": "cloud"} for q in questions[:N_PER_DOMAIN]]


def gen_tourism(rng):
    cities = ["Barcelona", "Tokyo", "Paris", "Rome", "Lisbon", "Bangkok",
              "Sydney", "Cairo", "Vienna", "Prague", "Athens", "Seoul",
              "Amsterdam", "Istanbul", "Dublin", "Oslo"]
    friends = ["Alice", "Bob", "Emma", "Liam", "Olivia", "Noah", "Sophia",
               "James", "Mia", "Lucas", "Ava", "Ethan"]
    months = ["January", "March", "April", "June", "August", "September",
              "October", "December"]
    durations = ["5-day", "6-day", "7-day", "10-day", "two-week", "4-day"]
    rows = []
    for _ in range(N_PER_DOMAIN):
        city = rng.choice(cities)
        friend = rng.choice(friends)
        month = rng.choice(months)
        dur = rng.choice(durations)
        budget = rng.randint(700, 5000)
        prompt = (f"I'm planning a {dur} trip to {city} with my friend {friend} "
                  f"in {month}. My budget is {budget} USD. Please help me plan my itinerary.")
        rows.append({"prompt": prompt, "route": "collaborative"})
    return rows


def gen_banking(rng):
    names = ["John Smith", "Robert Lee", "Maria Garcia", "David Chen",
             "Sarah Johnson", "Michael Brown", "Linda Davis", "James Wilson",
             "Patricia Moore", "William Taylor", "Karen Anderson", "Thomas Martin"]
    rows = []
    templates = [
        ("My name is {name} and my credit card number is {cc}. I noticed a charge "
         "of {amt} USD on my account {acct}. Can you help me dispute it?"),
        ("I am {name}. My card {cc} was charged {amt} dollars and my bank account "
         "{acct} shows a wrong balance. What should I do?"),
        ("Hello, this is {name}. Please review the {amt} USD transaction on card "
         "{cc} linked to account {acct}; I did not authorize it."),
    ]
    for _ in range(N_PER_DOMAIN):
        name = rng.choice(names)
        cc = make_card(rng)
        amt = rng.randint(50, 4000)
        acct = "".join(str(rng.randint(0, 9)) for _ in range(10))
        prompt = rng.choice(templates).format(name=name, cc=cc, amt=amt, acct=acct)
        rows.append({"prompt": prompt, "route": "edge"})
    return rows


def main():
    rng = random.Random(SEED)

    # Authentic medical prompts shipped with the repo.
    med_path = os.path.join(os.path.dirname(__file__), "..", "Dataset", "prism_dataset.xlsx")
    med_df = pd.read_excel(med_path)
    medical = [{"prompt": p, "route": "collaborative"} for p in med_df["prompt"].tolist()]

    rows = []
    for prefix, items in [("C", gen_common(rng)),
                          ("T", gen_tourism(rng)),
                          ("M", medical),
                          ("B", gen_banking(rng))]:
        for i, r in enumerate(items, 1):
            rows.append({"id": f"{prefix}{i}", "prompt": r["prompt"], "route": r["route"]})

    out = pd.DataFrame(rows, columns=["id", "prompt", "route"])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_excel(OUT_PATH, index=False)
    print(f"Wrote {len(out)} prompts to {os.path.normpath(OUT_PATH)}")
    print(out.groupby("route").size().to_string())


if __name__ == "__main__":
    main()
