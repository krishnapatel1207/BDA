# backend/generate_data.py
import os, csv, random

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT = os.path.join(OUT_DIR, "patients.csv")
os.makedirs(OUT_DIR, exist_ok=True)

def generate_csv(path=OUT, n=10000, seed=42):
    random.seed(seed)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Age","Height","Weight","HbA1c","Activity","LDL","Risk"])
        for _ in range(n):
            age = random.randint(18, 85)
            height = random.randint(140, 195)
            weight = random.randint(45, 140)
            hba1c = round(random.uniform(4.5, 10.5), 2)
            activity = random.randint(0, 600)      # minutes/week
            ldl = random.randint(60, 260)
            bmi = weight / ((height/100)**2)
            # Create multiclass risk label 0=Low,1=Medium,2=High with reasonable rules
            if hba1c > 8.0 or ldl > 190 or bmi > 35:
                risk = 2
            elif hba1c > 6.5 or ldl > 160 or bmi > 30 or age > 65:
                risk = 1
            else:
                risk = 0
            writer.writerow([age, height, weight, hba1c, activity, ldl, risk])
    print(f"Generated {path} with {n} rows")

if __name__ == "__main__":
    generate_csv()
