#!/usr/bin/env python3
"""
Convert Markdown Documentation to PDF
Students: 2024ab05134, 2024aa05664

This script converts the comprehensive pipeline documentation from Markdown to PDF format.
Requires: pip install markdown2 pdfkit weasyprint (choose one PDF engine)
"""

import os
import sys
from pathlib import Path

def convert_with_weasyprint():
    """Convert using WeasyPrint (recommended)."""
    try:
        import markdown2
        from weasyprint import HTML, CSS
        
        # Read markdown file
        md_file = Path("END_TO_END_PIPELINE_DOCUMENTATION.md")
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown2.markdown(md_content, extras=['fenced-code-blocks', 'tables'])
        
        # Add CSS styling for professional appearance
        css_style = """
        @page {
            size: A4;
            margin: 2cm;
        }
        
        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            font-size: 2.5em;
        }
        
        h2 {
            color: #34495e;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
            margin-top: 30px;
            font-size: 2em;
        }
        
        h3 {
            color: #2c3e50;
            margin-top: 25px;
            font-size: 1.5em;
        }
        
        code {
            background-color: #f8f9fa;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #3498db;
            border-radius: 5px;
            overflow-x: auto;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        
        .page-break {
            page-break-before: always;
        }
        """
        
        # Create complete HTML document
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>End-to-End Data Management Pipeline Documentation</title>
            <style>{css_style}</style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Convert to PDF
        pdf_file = Path("END_TO_END_PIPELINE_DOCUMENTATION.pdf")
        HTML(string=html_doc).write_pdf(pdf_file)
        
        print(f"PDF successfully created: {pdf_file}")
        print(f"File size: {pdf_file.stat().st_size / 1024:.1f} KB")
        return True
        
    except ImportError as e:
        print(f"WeasyPrint not available: {e}")
        return False
    except Exception as e:
        print(f"Error with WeasyPrint conversion: {e}")
        return False

def convert_with_pdfkit():
    """Convert using pdfkit (alternative method)."""
    try:
        import markdown2
        import pdfkit
        
        # Read markdown file
        md_file = Path("END_TO_END_PIPELINE_DOCUMENTATION.md")
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown2.markdown(md_content, extras=['fenced-code-blocks', 'tables'])
        
        # PDF options for professional formatting
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None
        }
        
        # Convert to PDF
        pdf_file = "END_TO_END_PIPELINE_DOCUMENTATION.pdf"
        pdfkit.from_string(html_content, pdf_file, options=options)
        
        print(f"PDF successfully created: {pdf_file}")
        file_size = Path(pdf_file).stat().st_size / 1024
        print(f"File size: {file_size:.1f} KB")
        return True
        
    except ImportError as e:
        print(f"pdfkit not available: {e}")
        return False
    except Exception as e:
        print(f"Error with pdfkit conversion: {e}")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("Installing PDF conversion dependencies...")
    print("Choose one of the following installation methods:")
    print("")
    print("Option 1 - WeasyPrint (Recommended):")
    print("  pip install markdown2 weasyprint")
    print("")
    print("Option 2 - pdfkit:")
    print("  pip install markdown2 pdfkit")
    print("  # Also requires wkhtmltopdf system installation")
    print("")
    print("Option 3 - Simple HTML output:")
    print("  pip install markdown2")
    print("  # Then open the HTML file in browser and print to PDF")

def convert_to_html_only():
    """Fallback: Convert to HTML that can be printed to PDF."""
    try:
        import markdown2
        
        # Read markdown file
        md_file = Path("END_TO_END_PIPELINE_DOCUMENTATION.md")
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown2.markdown(md_content, extras=['fenced-code-blocks', 'tables'])
        
        # Add professional CSS styling
        css_style = """
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; margin-top: 30px; }
            h3 { color: #2c3e50; margin-top: 25px; }
            code { background-color: #f8f9fa; padding: 2px 5px; border-radius: 3px; }
            pre { background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; overflow-x: auto; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #3498db; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
        </style>
        """
        
        # Create complete HTML document
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>End-to-End Data Management Pipeline Documentation</title>
            {css_style}
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Save HTML file
        html_file = Path("END_TO_END_PIPELINE_DOCUMENTATION.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_doc)
        
        print(f"HTML file created: {html_file}")
        print("To convert to PDF:")
        print("1. Open the HTML file in your web browser")
        print("2. Press Ctrl+P (or Cmd+P on Mac)")
        print("3. Choose 'Save as PDF' or 'Print to PDF'")
        print("4. Adjust margins and settings as needed")
        return True
        
    except ImportError:
        print("markdown2 not available. Install with: pip install markdown2")
        return False
    except Exception as e:
        print(f"Error creating HTML: {e}")
        return False

def main():
    """Main conversion function."""
    print("END-TO-END PIPELINE DOCUMENTATION - PDF CONVERTER")
    print("=" * 60)
    print("Students: 2024ab05134, 2024aa05664")
    print("")
    
    # Check if markdown file exists
    md_file = Path("END_TO_END_PIPELINE_DOCUMENTATION.md")
    if not md_file.exists():
        print(f"Error: {md_file} not found!")
        return
    
    print(f"Converting: {md_file}")
    print(f"File size: {md_file.stat().st_size / 1024:.1f} KB")
    print("")
    
    # Try different conversion methods
    success = False
    
    print("Attempting WeasyPrint conversion...")
    success = convert_with_weasyprint()
    
    if not success:
        print("Attempting pdfkit conversion...")
        success = convert_with_pdfkit()
    
    if not success:
        print("PDF libraries not available. Creating HTML version...")
        success = convert_to_html_only()
    
    if not success:
        print("Conversion failed. Installing dependencies:")
        install_dependencies()
    else:
        print("")
        print("✅ Conversion completed successfully!")
        print("📄 Your comprehensive pipeline documentation is ready!")

if __name__ == "__main__":
    main()
