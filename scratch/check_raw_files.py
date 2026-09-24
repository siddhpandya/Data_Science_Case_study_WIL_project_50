from pathlib import Path

raw_dir = Path("data/raw")
expected = [
    "Applying for an ABN _ Australian Business Register.pdf",
    "Bulk billing - Medicare - Services Australia.pdf",
    "Check visa details and conditions.pdf",
    "Check visa details and conditions2.pdf",
    "Enrolling in Medicare - Medicare - Services Australia.pdf",
    "Fair Work system - Fair Work Ombudsman.pdf",
    "How your Medicare card and account work - Medicare - Services Australia.pdf",
    "I'm a migrant worker being treated unfairly - Fair Work Ombudsman.pdf",
    "I'm not getting pay slips - Fair Work Ombudsman.pdf",
    "My pay doesn't seem right - Fair Work Ombudsman.pdf",
    "Orientation - RMIT University.pdf",
    "Tax in Australia_ what you need to know _ Australian Taxation Office.pdf",
    "Tickets and payments - Transport Victoria.pdf",
    "Your first week in Australia _ Study Australia.pdf",
    "myki _ Your ticket to travel _ Public transport in Victoria - Transport Victoria.pdf",
    "home_affairs_student_visa_work_hours_20230831_current.html",
    "home_affairs_student_visa_work_hours_20220607_superseded.html"
]

missing = [f for f in expected if not (raw_dir / f).exists()]
print("Missing files:", missing)
print(f"Found {len(expected) - len(missing)} of {len(expected)} expected raw files.")
