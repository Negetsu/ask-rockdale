import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
import json
import time
import re
from urllib.parse import urljoin, urlparse, parse_qs
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import logging
from pathlib import Path
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('municode_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class OrdinanceSection:
    """Structure for storing ordinance data"""
    chapter: str
    section_id: str
    title: str
    content: str
    url: str
    node_id: str
    chapter_number: Optional[str] = None
    article: Optional[str] = None
    subsections: Optional[List[Dict]] = None
    municipality: str = "Rockdale County"
    scraped_at: Optional[str] = None

class RockdaleMunicodeScraper:
    """
    Targeted scraper for Rockdale County Municode site
    Uses known URL patterns and improved content extraction
    """
    
    def __init__(self, base_url="https://library.municode.com/ga/rockdale_county/codes/code_of_ordinances"):
        self.base_url = base_url
        self.driver = None
        self.ordinances = []
        self.failed_urls = []
        self.retry_count = 3
        self.delay_between_requests = 3  # Increased delay
        
        # Known Rockdale County ordinance patterns from search results
        self.known_chapters = [
            ("SPAGEOR_CH18AN", "Chapter 18 - ANIMALS"),
            ("SPAGEOR_CH42EN", "Chapter 42 - ENVIRONMENT"), 
            ("SPBPLDE_TIT2LAUSZO_CH222OREPAST", "Chapter 222 - OFF-STREET PARKING STANDARDS"),
            ("SPBPLDE_TIT1AD", "TITLE 1 - ADMINISTRATION"),
            ("SPBPLDE_TIT2LAUSZO_CH218USRE", "Chapter 218 - USE REGULATIONS"),
            ("SPBPLDE_TIT2LAUSZO_CH206BAZODI", "Chapter 206 - BASE ZONING DISTRICTS"),
            ("PTIRELA", "PART I - RELATED LAWS")
        ]
        
        # Create output directory
        self.output_dir = Path("scraped_data")
        self.output_dir.mkdir(exist_ok=True)
        
    def setup_driver(self, headless=True):
        """Initialize Selenium WebDriver with robust configuration"""
        try:
            chrome_options = Options()
            
            if headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            # Additional options for stability
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--allow-running-insecure-content")
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.implicitly_wait(15)
            
            # Remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logger.info("Chrome WebDriver initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            return False
    
    def wait_for_content_load(self, timeout=20):
        """Wait for Municode content to fully load"""
        try:
            # Wait for any of these elements that indicate content has loaded
            content_indicators = [
                "//div[contains(@class, 'content')]",
                "//div[contains(@class, 'section')]", 
                "//div[contains(@class, 'ordinance')]",
                "//div[contains(@class, 'code')]",
                "//p[contains(text(), 'Sec.') or contains(text(), 'Section')]",
                "//h1 | //h2 | //h3"
            ]
            
            for indicator in content_indicators:
                try:
                    WebDriverWait(self.driver, timeout).until(
                        EC.presence_of_element_located((By.XPATH, indicator))
                    )
                    logger.info(f"Content loaded - found element: {indicator}")
                    return True
                except TimeoutException:
                    continue
            
            logger.warning("No content indicators found within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for content: {e}")
            return False
    
    def extract_content_from_page(self, url):
        """Extract all available content from a Municode page"""
        try:
            logger.info(f"Loading page: {url}")
            self.driver.get(url)
            
            # Wait for content to load
            if not self.wait_for_content_load():
                logger.warning(f"Content may not have fully loaded for {url}")
            
            # Additional wait for JavaScript
            time.sleep(5)
            
            # Get page source and parse with BeautifulSoup for better text extraction
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Extract title
            title = self.extract_page_title(soup)
            
            # Extract main content
            content = self.extract_page_content(soup)
            
            # Extract any section structure
            sections = self.extract_sections(soup)
            
            # Get node_id from URL
            node_id = self.extract_node_id(url)
            
            return {
                "title": title,
                "content": content,
                "sections": sections,
                "node_id": node_id,
                "url": url
            }
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    def extract_page_title(self, soup):
        """Extract page title using multiple strategies"""
        # Try different title selectors
        title_selectors = [
            'h1',
            '.page-title',
            '.section-title', 
            '.ordinance-title',
            'title'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element and element.get_text(strip=True):
                title = element.get_text(strip=True)
                # Clean up title
                title = re.sub(r'\s+', ' ', title)
                if len(title) > 10:  # Reasonable title length
                    return title
        
        return "Unknown Title"
    
    def extract_page_content(self, soup):
        """Extract main content from page"""
        content_parts = []
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Try different content extraction strategies
        content_selectors = [
            '.content',
            '.main-content',
            '.ordinance-content',
            '.section-content',
            '[role="main"]',
            'main'
        ]
        
        # First try specific content containers
        for selector in content_selectors:
            content_container = soup.select_one(selector)
            if content_container:
                text = content_container.get_text(separator='\n', strip=True)
                if len(text) > 100:  # Substantial content
                    content_parts.append(text)
                    break
        
        # If no specific container found, extract from paragraphs and divs
        if not content_parts:
            for element in soup.find_all(['p', 'div'], string=True):
                text = element.get_text(strip=True)
                if (len(text) > 50 and 
                    not any(skip in text.lower() for skip in ['municode', 'next', 'search', 'navigation'])):
                    content_parts.append(text)
        
        # Clean and join content
        if content_parts:
            content = '\n\n'.join(content_parts)
            # Clean up whitespace
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = re.sub(r' +', ' ', content)
            return content.strip()
        
        return ""
    
    def extract_sections(self, soup):
        """Extract individual sections from the page"""
        sections = []
        
        # Look for section patterns
        section_patterns = [
            r'Sec\.\s*[\d-]+\.',
            r'Section\s+[\d-]+',
            r'§\s*[\d-]+',
            r'\d+-\d+'
        ]
        
        # Find elements that look like section headers
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'div']):
            text = element.get_text(strip=True)
            
            for pattern in section_patterns:
                if re.search(pattern, text):
                    # This looks like a section header
                    section_content = self.extract_section_content(element)
                    if section_content:
                        sections.append({
                            'title': text,
                            'content': section_content
                        })
                    break
        
        return sections if sections else None
    
    def extract_section_content(self, header_element):
        """Extract content following a section header"""
        content_parts = []
        current = header_element.next_sibling
        
        # Collect content until next header or end
        while current and len(content_parts) < 10:  # Reasonable limit
            if hasattr(current, 'get_text'):
                text = current.get_text(strip=True)
                if text and len(text) > 20:
                    content_parts.append(text)
                    
                # Stop if we hit another section header
                if re.search(r'Sec\.\s*[\d-]+\.|Section\s+[\d-]+', text):
                    break
            
            current = current.next_sibling
        
        return '\n\n'.join(content_parts) if content_parts else ""
    
    def extract_node_id(self, url):
        """Extract nodeId from URL"""
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            return query_params.get('nodeId', [''])[0]
        except:
            return ""
    
    def discover_additional_sections(self, base_node_id):
        """Try to discover related sections by modifying node IDs"""
        discovered_urls = []
        
        # Common section suffixes for Municode
        common_suffixes = [
            '_S1', '_S2', '_S3', '_S4', '_S5',
            '_ART1', '_ART2', '_ART3',
            '_ARTII', '_ARTIII', '_ARTIV',
            '_DE', '_GE', '_PE', '_VI'
        ]
        
        for suffix in common_suffixes:
            test_node_id = base_node_id + suffix
            test_url = f"{self.base_url}?nodeId={test_node_id}"
            discovered_urls.append(test_url)
        
        return discovered_urls
    
    def scrape_known_chapters(self):
        """Scrape all known chapters and their sections"""
        if not self.driver:
            if not self.setup_driver():
                return []
        
        all_urls_to_scrape = []
        
        # Add known chapter URLs
        for node_id, chapter_name in self.known_chapters:
            url = f"{self.base_url}?nodeId={node_id}"
            all_urls_to_scrape.append((url, chapter_name, node_id))
            
            # Try to discover related sections
            discovered = self.discover_additional_sections(node_id)
            for disc_url in discovered:
                disc_node_id = self.extract_node_id(disc_url)
                all_urls_to_scrape.append((disc_url, f"{chapter_name} - Related", disc_node_id))
        
        logger.info(f"Total URLs to scrape: {len(all_urls_to_scrape)}")
        
        # Scrape each URL
        for i, (url, chapter_name, node_id) in enumerate(all_urls_to_scrape):
            logger.info(f"Processing {i+1}/{len(all_urls_to_scrape)}: {chapter_name}")
            
            content_data = self.extract_content_from_page(url)
            
            if content_data and content_data['content']:
                # Parse chapter info
                chapter_num = self.extract_chapter_number(chapter_name)
                
                ordinance = OrdinanceSection(
                    chapter=chapter_name,
                    chapter_number=chapter_num,
                    section_id=node_id,
                    title=content_data['title'],
                    content=content_data['content'],
                    url=url,
                    node_id=node_id,
                    subsections=content_data['sections'],
                    scraped_at=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                self.ordinances.append(ordinance)
                logger.info(f"Successfully scraped: {chapter_name}")
            else:
                logger.warning(f"No content found for: {url}")
                self.failed_urls.append(url)
            
            # Be respectful to the server
            time.sleep(self.delay_between_requests)
        
        logger.info(f"Scraping completed. Found {len(self.ordinances)} ordinance sections.")
        return self.ordinances
    
    def extract_chapter_number(self, chapter_name):
        """Extract chapter number from chapter name"""
        match = re.search(r'Chapter\s+(\d+)', chapter_name)
        if match:
            return match.group(1)
        
        match = re.search(r'TITLE\s+(\d+)', chapter_name)
        if match:
            return f"T{match.group(1)}"
        
        return None
    
    def save_to_json(self, filename=None):
        """Save scraped ordinances to JSON file"""
        if filename is None:
            filename = self.output_dir / "rockdale_ordinances.json"
        
        ordinances_dict = [asdict(ord) for ord in self.ordinances]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(ordinances_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Ordinances saved to {filename}")
    
    def save_to_csv(self, filename=None):
        """Save scraped ordinances to CSV file"""
        if filename is None:
            filename = self.output_dir / "rockdale_ordinances.csv"
        
        if not self.ordinances:
            logger.warning("No ordinances to save")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            writer.writerow([
                'Chapter', 'Chapter Number', 'Section ID', 'Title', 
                'Content', 'URL', 'Node ID', 'Municipality', 'Scraped At'
            ])
            
            for ord in self.ordinances:
                writer.writerow([
                    ord.chapter, ord.chapter_number, ord.section_id,
                    ord.title, ord.content, ord.url, ord.node_id,
                    ord.municipality, ord.scraped_at
                ])
        
        logger.info(f"Ordinances saved to {filename}")
    
    def save_summary_report(self, filename=None):
        """Save a summary report of scraped content"""
        if filename is None:
            filename = self.output_dir / "scraping_summary.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("ROCKDALE COUNTY ORDINANCE SCRAPING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total ordinances scraped: {len(self.ordinances)}\n")
            f.write(f"Failed URLs: {len(self.failed_urls)}\n")
            f.write(f"Scraping completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if self.ordinances:
                f.write("SCRAPED CHAPTERS:\n")
                f.write("-" * 20 + "\n")
                chapters = {}
                for ord in self.ordinances:
                    if ord.chapter not in chapters:
                        chapters[ord.chapter] = 0
                    chapters[ord.chapter] += 1
                
                for chapter, count in chapters.items():
                    f.write(f"{chapter}: {count} sections\n")
                
                f.write(f"\nSAMPLE CONTENT:\n")
                f.write("-" * 15 + "\n")
                sample = self.ordinances[0]
                f.write(f"Chapter: {sample.chapter}\n")
                f.write(f"Title: {sample.title}\n")
                f.write(f"URL: {sample.url}\n")
                f.write(f"Content preview: {sample.content[:300]}...\n\n")
            
            if self.failed_urls:
                f.write("FAILED URLS:\n")
                f.write("-" * 12 + "\n")
                for url in self.failed_urls:
                    f.write(f"{url}\n")
        
        logger.info(f"Summary report saved to {filename}")
    
    def cleanup(self):
        """Close the browser driver and clean up"""
        if self.driver:
            self.driver.quit()
            logger.info("Browser driver closed")

# Example usage and main execution
if __name__ == "__main__":
    # Initialize scraper with longer delays for stability
    scraper = RockdaleMunicodeScraper()
    scraper.delay_between_requests = 4  # Slower scraping for stability
    
    try:
        # Scrape known chapters
        ordinances = scraper.scrape_known_chapters()
        
        # Save results in multiple formats
        scraper.save_to_json()
        scraper.save_to_csv() 
        scraper.save_summary_report()
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"SCRAPING SUMMARY")
        print(f"{'='*50}")
        print(f"Total ordinances scraped: {len(ordinances)}")
        print(f"Failed URLs: {len(scraper.failed_urls)}")
        
        if ordinances:
            print(f"\nChapters found:")
            chapters = set(ord.chapter for ord in ordinances)
            for chapter in sorted(chapters):
                count = sum(1 for ord in ordinances if ord.chapter == chapter)
                print(f"  - {chapter}: {count} sections")
            
            print(f"\nSample ordinance:")
            sample = ordinances[0]
            print(f"Chapter: {sample.chapter}")
            print(f"Title: {sample.title}")
            print(f"URL: {sample.url}")
            print(f"Content preview: {sample.content[:200]}...")
        
        if scraper.failed_urls:
            print(f"\nFailed URLs: {len(scraper.failed_urls)}")
        
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.cleanup()