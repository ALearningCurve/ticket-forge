# scrape_github_issues.py
from github import Github
import pandas as pd
import time
import os
from dotenv import load_dotenv

# YOUR WORKING TOKEN
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN missing. Add it to .env file.")

g = Github(GITHUB_TOKEN)
print(f"✅ Authenticated as: {g.get_user().login}\n")

# Repos to scrape
REPOS = [
    "hashicorp/terraform",
    "ansible/ansible",
    "prometheus/prometheus"
]

def scrape_repo_issues(repo_name, max_issues=300):
    """Scrape issues from a GitHub repo"""
    print(f"📦 Scraping {repo_name}...")
    
    try:
        repo = g.get_repo(repo_name)
        issues_data = []
        
        # Get closed issues
        issues = repo.get_issues(state='closed', sort='created', direction='desc')
        
        count = 0
        for issue in issues:
            # Skip pull requests
            if issue.pull_request:
                continue
            
            # Skip issues with no body
            if not issue.body or len(issue.body) < 50:
                continue
            
            # Get labels
            labels = [label.name for label in issue.labels]
            if not labels:
                continue
            
            # Determine complexity
            complexity = 'simple'
            if issue.comments > 3 or len(issue.body) > 500:
                complexity = 'medium'
            if issue.comments > 10 or len(issue.body) > 1500:
                complexity = 'complex'
            
            # Calculate base effort
            if issue.closed_at and issue.created_at:
                days_open = (issue.closed_at - issue.created_at).days
                base_effort = min(days_open * 1.5, 30)  # Cap at 30 hours
            else:
                base_effort = {'simple': 3, 'medium': 6, 'complex': 12}[complexity]
            
            issues_data.append({
                'id': f"{repo_name.split('/')[1].upper()}-{issue.number}",
                'repo': repo_name,
                'title': issue.title,
                'description': issue.body[:1500],  # Truncate long descriptions
                'labels': ','.join(labels[:5]),  # Top 5 labels
                'complexity': complexity,
                'created_at': issue.created_at.isoformat(),
                'closed_at': issue.closed_at.isoformat() if issue.closed_at else None,
                'comments_count': issue.comments,
                'base_effort_hours': round(base_effort, 1),
                'url': issue.html_url
            })
            
            count += 1
            
            if count % 50 == 0:
                print(f"  ✓ Scraped {count} issues...")
            
            if count >= max_issues:
                break
            
            # Be nice to GitHub API
            time.sleep(0.8)
        
        print(f"  ✅ Got {len(issues_data)} issues from {repo_name}\n")
        return issues_data
        
    except Exception as e:
        print(f"  ❌ Error scraping {repo_name}: {e}\n")
        return []

# Main scraping
print("🚀 Starting GitHub Issues Scraper")
print("=" * 50)

all_issues = []

for repo in REPOS:
    repo_issues = scrape_repo_issues(repo, max_issues=300)
    all_issues.extend(repo_issues)
    
    # Save progress after each repo
    if all_issues:
        df_temp = pd.DataFrame(all_issues)
        df_temp.to_csv('tickets_partial.csv', index=False)
        print(f"💾 Progress saved: {len(all_issues)} total issues so far\n")

# Final save
if all_issues:
    df = pd.DataFrame(all_issues)
    df.to_csv('tickets_raw.csv', index=False)
    
    print("=" * 50)
    print(f"🎉 SCRAPING COMPLETE!")
    print(f"📊 Total issues scraped: {len(all_issues)}")
    print(f"\n📊 Breakdown by repo:")
    print(df['repo'].value_counts())
    print(f"\n📊 Complexity distribution:")
    print(df['complexity'].value_counts())
    print(f"\n💾 Saved to: tickets_raw.csv")
else:
    print("❌ No issues scraped")