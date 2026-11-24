import unittest

from coded_tools.knowledge.workshop_knowledge import WorkshopKnowledgeTool


class WorkshopKnowledgeToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_the_workshop_narrative(self):
        text = await WorkshopKnowledgeTool().async_invoke({}, {})

        self.assertIn("Focus Question", text)
        self.assertIn("This is a journey for all of us", text)


if __name__ == "__main__":
    unittest.main()
