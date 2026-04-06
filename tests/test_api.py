import unittest

from fastapi.testclient import TestClient

from app import _SESSIONS, app


class APITests(unittest.TestCase):
    def setUp(self) -> None:
        _SESSIONS.clear()
        self.client = TestClient(app)

    def test_root_and_health_endpoints(self) -> None:
        root_response = self.client.get("/")
        health_response = self.client.get("/health")

        self.assertEqual(root_response.status_code, 200)
        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(health_response.json()["status"], "ok")

    def test_reset_step_grade_round_trip(self) -> None:
        reset_response = self.client.post("/reset", json={"scenario_name": "easy"})
        self.assertEqual(reset_response.status_code, 200)

        payload = reset_response.json()
        session_id = payload["session_id"]
        current_ticket = payload["observation"]["current_ticket"]

        step_response = self.client.post(
            f"/step/{session_id}",
            json={
                "ticket_id": current_ticket["id"],
                "department": current_ticket["category_hint"],
                "priority": "high" if current_ticket["urgency"] >= 4 else "medium",
                "action_type": "escalate" if current_ticket["urgency"] >= 4 else "resolve",
            },
        )
        self.assertEqual(step_response.status_code, 200)

        grade_response = self.client.get(f"/grade/{session_id}")
        self.assertEqual(grade_response.status_code, 200)
        self.assertIn("grade", grade_response.json())


if __name__ == "__main__":
    unittest.main()
