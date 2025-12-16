import os
from app.crawler.parser import NJUParser

def test_parser():
    # Test List Page
    list_file = "list_page.html"
    if os.path.exists(list_file):
        print(f"Reading {list_file}...")
        with open(list_file, "r", encoding="utf-8") as f:
            html = f.read()
        
        print(f"Parsing list page (Length: {len(html)})...")
        articles = NJUParser.parse_list_page(html, "https://is.nju.edu.cn/57162/list.htm")
        
        print(f"Found {len(articles)} articles.")
        if articles:
            first = articles[0]
            print("First article sample:")
            print(f"  Title: {first['title']}")
            print(f"  Date: {first['date']}")
            print(f"  URL: {first['url']}")
            print(f"  Is Top: {first['is_top']}")
    else:
        print(f"{list_file} not found.")

    # Test Detail Page
    detail_file = "detail_page.html"
    if os.path.exists(detail_file):
        print(f"\nReading {detail_file}...")
        with open(detail_file, "r", encoding="utf-8") as f:
            detail_html = f.read()
        
        detail = NJUParser.parse_detail_page(detail_html)
        print("Detail page sample:")
        print(f"  Title: {detail['title']}")
        print(f"  Date (from meta): {detail['publish_date']}")
        print(f"  Content Length: {len(detail['content_html'])}")
    else:
        print(f"{detail_file} not found.")

if __name__ == "__main__":
    test_parser()
