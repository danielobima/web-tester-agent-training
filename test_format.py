import asyncio
import json
from format_mind2web import process_aria_snapshot, format_task_steps
from playwright.async_api import async_playwright

async def test_format():
    dummy_html = """
    <html>
    <body>
        <h1>Test Goal</h1>
        <button id="btn1">Click Me</button>
        <input type="text" placeholder="Type here">
        <a href="#">Link</a>
    </body>
    </html>
    """
    
    sample = {
        'cleaned_html': dummy_html,
        'confirmed_task': 'Test Goal',
        'action_reprs': 'CLICK on Click Me',
        'annotation_id': 'task1',
        'step_id': 0,
        'website': 'test.com',
        'pos_candidates': [
            {
                'role': 'button',
                'attributes': {'name': 'Click Me'},
                'backend_node_id': '123'
            }
        ]
    }
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        formatted_samples = await format_task_steps('task1', [sample], page)
        print("Formatted Samples:")
        for sample in formatted_samples:
            print("--- INPUT ---")
            print(json.dumps(sample['input'], indent=2))
            print("--- OUTPUT ---")
            print(json.dumps(sample['output'], indent=2))
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_format())
