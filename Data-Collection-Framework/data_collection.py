# Real-World Social Network Data Collection Framework
import requests
import json
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import time
import sqlite3
import hashlib
import os
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NetworkData:
    """Container for network data with metadata"""
    nodes: pd.DataFrame
    edges: pd.DataFrame
    metadata: Dict
    collection_date: datetime
    source: str

class EthicalDataCollector:
    """
    Ethical data collection framework following research guidelines
    """
    
    def __init__(self, storage_path="./data", rate_limit=1.0):
        self.storage_path = storage_path
        self.rate_limit = rate_limit  # Requests per second
        self.session = requests.Session()
        self.last_request_time = 0
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize database for caching
        self.init_database()
        
        logger.info(f"EthicalDataCollector initialized with rate limit: {rate_limit} req/s")
    
    def init_database(self):
        """Initialize SQLite database for data caching"""
        self.db_path = os.path.join(self.storage_path, "network_cache.db")
        conn = sqlite3.connect(self.db_path)
        
        # Create tables
        conn.execute('''
            CREATE TABLE IF NOT EXISTS network_data (
                id TEXT PRIMARY KEY,
                source TEXT,
                collection_date TEXT,
                data_hash TEXT,
                metadata TEXT,
                UNIQUE(source, data_hash)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                network_id TEXT,
                node_id TEXT,
                features TEXT,
                label TEXT,
                FOREIGN KEY(network_id) REFERENCES network_data(id)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                network_id TEXT,
                source_id TEXT,
                target_id TEXT,
                weight REAL,
                attributes TEXT,
                FOREIGN KEY(network_id) REFERENCES network_data(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def respect_rate_limit(self):
        """Ensure we respect API rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def save_network_data(self, network_data: NetworkData) -> str:
        """Save network data to database"""
        network_id = hashlib.md5(
            f"{network_data.source}_{network_data.collection_date}".encode()
        ).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        
        # Save network metadata
        conn.execute('''
            INSERT OR REPLACE INTO network_data 
            (id, source, collection_date, data_hash, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            network_id,
            network_data.source,
            network_data.collection_date.isoformat(),
            hashlib.md5(str(network_data.metadata).encode()).hexdigest(),
            json.dumps(network_data.metadata)
        ))
        
        # Save nodes
        for _, node in network_data.nodes.iterrows():
            conn.execute('''
                INSERT OR REPLACE INTO nodes 
                (network_id, node_id, features, label)
                VALUES (?, ?, ?, ?)
            ''', (
                network_id,
                str(node.get('user_id', node.name)),
                json.dumps(node.to_dict()),
                str(node.get('influence_level', ''))
            ))
        
        # Save edges
        for _, edge in network_data.edges.iterrows():
            conn.execute('''
                INSERT OR REPLACE INTO edges 
                (network_id, source_id, target_id, weight, attributes)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                network_id,
                str(edge['source']),
                str(edge['target']),
                float(edge.get('weight', 1.0)),
                json.dumps(edge.to_dict())
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Network data saved with ID: {network_id}")
        return network_id

class GitHubCollaborationNetworkCollector(EthicalDataCollector):
    """
    Collect collaboration networks from GitHub (using public API)
    """
    
    def __init__(self, github_token = "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", **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Academic-Research-Bot/1.0"
        }
        
        if github_token:
            self.headers["Authorization"] = f"token {github_token}"
            logger.info("GitHub token provided - higher rate limits available")
        else:
            logger.warning("No GitHub token - limited to 60 requests/hour")
    
    def collect_repository_network(self, 
                                 repo_owner: str, 
                                 repo_name: str, 
                                 max_contributors: int = 100) -> NetworkData:
        """
        Collect collaboration network from a GitHub repository
        
        Args:
            repo_owner: Repository owner
            repo_name: Repository name
            max_contributors: Maximum number of contributors to analyze
            
        Returns:
            NetworkData object with collaboration network
        """
        logger.info(f"Collecting network for {repo_owner}/{repo_name}")
        
        # Get repository contributors
        contributors = self._get_contributors(repo_owner, repo_name, max_contributors)
        if not contributors:
            logger.error("No contributors found")
            return None
        
        # Build collaboration network based on co-contributions
        nodes_data = []
        edges_data = []
        
        # Create nodes (contributors)
        for contributor in contributors:
            self.respect_rate_limit()
            
            # Get contributor details
            user_data = self._get_user_details(contributor['login'])
            
            node_features = {
                'user_id': contributor['login'],
                'contributions': contributor['contributions'],
                'public_repos': user_data.get('public_repos', 0),
                'followers': user_data.get('followers', 0),
                'following': user_data.get('following', 0),
                'created_at': user_data.get('created_at', ''),
                'location': user_data.get('location', ''),
                'company': user_data.get('company', ''),
                'account_type': user_data.get('type', 'User')
            }
            
            nodes_data.append(node_features)
        
        # Create collaboration edges based on shared commits/issues
        logger.info("Building collaboration edges...")
        edges_data = self._build_collaboration_edges(repo_owner, repo_name, contributors)
        
        # Convert to DataFrames
        nodes_df = pd.DataFrame(nodes_data)
        edges_df = pd.DataFrame(edges_data)
        
        # Add derived features
        nodes_df = self._add_network_features(nodes_df, edges_df)
        
        # Create labels based on contribution levels
        nodes_df['influence_level'] = self._categorize_influence(nodes_df)
        
        metadata = {
            'repository': f"{repo_owner}/{repo_name}",
            'num_nodes': len(nodes_df),
            'num_edges': len(edges_df),
            'collection_method': 'github_api',
            'features': list(nodes_df.columns),
            'network_type': 'collaboration'
        }
        
        return NetworkData(
            nodes=nodes_df,
            edges=edges_df,
            metadata=metadata,
            collection_date=datetime.now(),
            source='github'
        )
    
    def _get_contributors(self, owner: str, repo: str, max_count: int) -> List[Dict]:
        """Get repository contributors"""
        self.respect_rate_limit()
        
        url = f"{self.base_url}/repos/{owner}/{repo}/contributors"
        params = {'per_page': min(100, max_count)}
        
        try:
            response = self.session.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            contributors = response.json()
            
            logger.info(f"Found {len(contributors)} contributors")
            return contributors[:max_count]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching contributors: {e}")
            return []
    
    def _get_user_details(self, username: str) -> Dict:
        """Get detailed user information"""
        url = f"{self.base_url}/users/{username}"
        
        try:
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching user {username}: {e}")
            return {}
    
    def _build_collaboration_edges(self, owner: str, repo: str, contributors: List[Dict]) -> List[Dict]:
        """Build edges based on collaboration patterns"""
        edges = []
        
        # Get commit data to find collaboration patterns
        commits = self._get_recent_commits(owner, repo, limit=200)
        
        # Build co-author network
        commit_authors = {}
        for commit in commits:
            if commit.get('author') and commit.get('committer'):
                author = commit['author'].get('login')
                committer = commit['committer'].get('login')
                
                if author and committer and author != committer:
                    key = tuple(sorted([author, committer]))
                    commit_authors[key] = commit_authors.get(key, 0) + 1
        
        # Convert to edge list
        for (user1, user2), weight in commit_authors.items():
            edges.append({
                'source': user1,
                'target': user2,
                'weight': weight,
                'relationship_type': 'co_contribution'
            })
        
        # Add follower relationships (if available)
        contributor_logins = [c['login'] for c in contributors]
        for contrib in contributors[:20]:  # Limit to avoid rate limits
            self.respect_rate_limit()
            followers = self._get_user_connections(contrib['login'], 'followers')
            following = self._get_user_connections(contrib['login'], 'following')
            
            # Add edges for mutual follows within contributor set
            for follower in followers:
                if follower in contributor_logins and follower != contrib['login']:
                    edges.append({
                        'source': follower,
                        'target': contrib['login'],
                        'weight': 1.0,
                        'relationship_type': 'follows'
                    })
        
        logger.info(f"Built {len(edges)} collaboration edges")
        return edges
    
    def _get_recent_commits(self, owner: str, repo: str, limit: int = 100) -> List[Dict]:
        """Get recent commits for collaboration analysis"""
        self.respect_rate_limit()
        
        url = f"{self.base_url}/repos/{owner}/{repo}/commits"
        params = {'per_page': min(100, limit)}
        
        try:
            response = self.session.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching commits: {e}")
            return []
    
    def _get_user_connections(self, username: str, connection_type: str) -> List[str]:
        """Get user's followers or following"""
        self.respect_rate_limit()
        
        url = f"{self.base_url}/users/{username}/{connection_type}"
        params = {'per_page': 50}  # Limit to avoid rate limits
        
        try:
            response = self.session.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            connections = response.json()
            
            return [conn['login'] for conn in connections]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {connection_type} for {username}: {e}")
            return []
    
    def _add_network_features(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
        """Add network-based features to nodes"""
        if edges_df.empty:
            nodes_df['degree'] = 0
            nodes_df['betweenness_centrality'] = 0
            nodes_df['closeness_centrality'] = 0
            return nodes_df
        
        # Create NetworkX graph
        G = nx.from_pandas_edgelist(edges_df, source='source', target='target', 
                                   edge_attr='weight', create_using=nx.Graph())
        
        # Calculate centrality measures
        centralities = {
            'degree': nx.degree_centrality(G),
            'betweenness_centrality': nx.betweenness_centrality(G),
            'closeness_centrality': nx.closeness_centrality(G)
        }
        
        # Add centrality features to nodes
        for measure, values in centralities.items():
            nodes_df[measure] = nodes_df['user_id'].map(values).fillna(0)
        
        return nodes_df
    
    def _categorize_influence(self, nodes_df: pd.DataFrame) -> List[str]:
        """Categorize nodes into influence levels based on network metrics"""
        if nodes_df.empty:
            return []
        
        labels = []
        for _, node in nodes_df.iterrows():
            contributions = node.get('contributions', 0)
            followers = node.get('followers', 0)
            degree = node.get('degree', 0)
            betweenness = node.get('betweenness_centrality', 0)
            
            # Score based on multiple factors
            influence_score = (
                np.log1p(contributions) * 0.3 +
                np.log1p(followers) * 0.2 +
                degree * 0.3 +
                betweenness * 0.2
            )
            
            # Categorize into levels
            if influence_score > 0.8:
                labels.append('high_influence')
            elif influence_score > 0.4:
                labels.append('medium_influence')
            else:
                labels.append('low_influence')
        
        return labels

class TwitterNetworkCollector(EthicalDataCollector):
    """
    Collect social networks from Twitter (using Academic API v2)
    Note: Requires Academic Research product track access
    """
    
    def __init__(self, bearer_token: str, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://api.twitter.com/2"
        self.headers = {
            "Authorization": f"Bearer {bearer_token}",
            "User-Agent": "Academic-Research-Bot/1.0"
        }
    
    def collect_hashtag_network(self, hashtag: str, max_tweets: int = 1000) -> NetworkData:
        """
        Collect network of users interacting around a hashtag
        
        Args:
            hashtag: Hashtag to analyze (without #)
            max_tweets: Maximum number of tweets to analyze
            
        Returns:
            NetworkData object with interaction network
        """
        logger.info(f"Collecting network for hashtag: #{hashtag}")
        
        # Search for tweets with the hashtag
        tweets = self._search_tweets(hashtag, max_tweets)
        if not tweets:
            logger.error("No tweets found")
            return None
        
        # Extract users and interactions
        nodes_data = []
        edges_data = []
        user_cache = {}
        
        for tweet in tweets:
            # Get tweet author
            author_id = tweet['author_id']
            if author_id not in user_cache:
                user_info = self._get_user_info(author_id)
                user_cache[author_id] = user_info
            
            author_info = user_cache[author_id]
            
            # Add author as node
            if not any(node['user_id'] == author_id for node in nodes_data):
                nodes_data.append({
                    'user_id': author_id,
                    'username': author_info.get('username', ''),
                    'name': author_info.get('name', ''),
                    'followers_count': author_info.get('public_metrics', {}).get('followers_count', 0),
                    'following_count': author_info.get('public_metrics', {}).get('following_count', 0),
                    'tweet_count': author_info.get('public_metrics', {}).get('tweet_count', 0),
                    'verified': author_info.get('verified', False),
                    'created_at': author_info.get('created_at', ''),
                    'location': author_info.get('location', ''),
                    'description': author_info.get('description', '')
                })
            
            # Process interactions (replies, retweets, mentions)
            if 'referenced_tweets' in tweet:
                for ref_tweet in tweet['referenced_tweets']:
                    if ref_tweet['type'] in ['replied_to', 'retweeted']:
                        # Get referenced tweet author
                        ref_tweet_data = self._get_tweet_info(ref_tweet['id'])
                        if ref_tweet_data and 'author_id' in ref_tweet_data:
                            target_id = ref_tweet_data['author_id']
                            
                            # Add edge
                            edges_data.append({
                                'source': author_id,
                                'target': target_id,
                                'weight': 1.0,
                                'interaction_type': ref_tweet['type']
                            })
            
            # Process mentions in tweet text
            if 'entities' in tweet and 'mentions' in tweet['entities']:
                for mention in tweet['entities']['mentions']:
                    mentioned_id = mention['id']
                    edges_data.append({
                        'source': author_id,
                        'target': mentioned_id,
                        'weight': 0.5,
                        'interaction_type': 'mention'
                    })
        
        # Convert to DataFrames
        nodes_df = pd.DataFrame(nodes_data)
        edges_df = pd.DataFrame(edges_data)
        
        # Add network features
        nodes_df = self._add_network_features(nodes_df, edges_df)
        
        # Create labels based on engagement levels
        nodes_df['engagement_level'] = self._categorize_engagement(nodes_df)
        
        metadata = {
            'hashtag': hashtag,
            'num_nodes': len(nodes_df),
            'num_edges': len(edges_df),
            'collection_method': 'twitter_api_v2',
            'features': list(nodes_df.columns),
            'network_type': 'social_interaction'
        }
        
        return NetworkData(
            nodes=nodes_df,
            edges=edges_df,
            metadata=metadata,
            collection_date=datetime.now(),
            source='twitter'
        )
    
    def _search_tweets(self, hashtag: str, max_results: int) -> List[Dict]:
        """Search for tweets containing hashtag"""
        self.respect_rate_limit()
        
        url = f"{self.base_url}/tweets/search/recent"
        params = {
            'query': f"#{hashtag} -is:retweet",
            'max_results': min(100, max_results),
            'tweet.fields': 'author_id,created_at,public_metrics,referenced_tweets,entities',
            'expansions': 'author_id,referenced_tweets.id'
        }
        
        try:
            response = self.session.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            tweets = data.get('data', [])
            logger.info(f"Found {len(tweets)} tweets")
            return tweets
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching tweets: {e}")
            return []
    
    def _get_user_info(self, user_id: str) -> Dict:
        """Get user information"""
        self.respect_rate_limit()
        
        url = f"{self.base_url}/users/{user_id}"
        params = {
            'user.fields': 'created_at,description,location,name,public_metrics,username,verified'
        }
        
        try:
            response = self.session.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            return data.get('data', {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching user {user_id}: {e}")
            return {}
    
    def _get_tweet_info(self, tweet_id: str) -> Dict:
        """Get tweet information"""
        self.respect_rate_limit()
        
        url = f"{self.base_url}/tweets/{tweet_id}"
        params = {
            'tweet.fields': 'author_id,created_at,public_metrics'
        }
        
        try:
            response = self.session.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            return data.get('data', {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching tweet {tweet_id}: {e}")
            return {}
    
    def _categorize_engagement(self, nodes_df: pd.DataFrame) -> List[str]:
        """Categorize nodes into engagement levels"""
        if nodes_df.empty:
            return []
        
        labels = []
        for _, node in nodes_df.iterrows():
            followers = node.get('followers_count', 0)
            tweets = node.get('tweet_count', 0)
            degree = node.get('degree', 0)
            
            # Score based on social metrics
            engagement_score = (
                np.log1p(followers) * 0.4 +
                np.log1p(tweets) * 0.2 +
                degree * 0.4
            )
            
            if engagement_score > 0.7:
                labels.append('high_engagement')
            elif engagement_score > 0.3:
                labels.append('medium_engagement')
            else:
                labels.append('low_engagement')
        
        return labels

class NetworkAnalyzer:
    """
    Analyze collected network data and extract insights
    """
    
    def __init__(self):
        pass
    
    def analyze_network_structure(self, network_data: NetworkData) -> Dict:
        """
        Analyze basic network structure properties
        
        Args:
            network_data: NetworkData object to analyze
            
        Returns:
            Dictionary with network analysis results
        """
        G = nx.from_pandas_edgelist(
            network_data.edges, 
            source='source', 
            target='target', 
            edge_attr='weight',
            create_using=nx.Graph()
        )
        
        analysis = {
            'basic_stats': {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'density': nx.density(G),
                'average_clustering': nx.average_clustering(G),
                'is_connected': nx.is_connected(G)
            },
            'connectivity': {
                'num_connected_components': nx.number_connected_components(G),
                'largest_component_size': len(max(nx.connected_components(G), key=len)) if G.nodes() else 0
            }
        }
        
        # Additional analysis for connected graphs
        if nx.is_connected(G) and G.number_of_nodes() > 1:
            analysis['connectivity']['diameter'] = nx.diameter(G)
            analysis['connectivity']['average_path_length'] = nx.average_shortest_path_length(G)
        
        # Degree distribution
        degrees = dict(G.degree())
        if degrees:
            degree_values = list(degrees.values())
            analysis['degree_distribution'] = {
                'mean_degree': np.mean(degree_values),
                'std_degree': np.std(degree_values),
                'max_degree': max(degree_values),
                'min_degree': min(degree_values)
            }
        
        # Community detection
        try:
            communities = nx.algorithms.community.greedy_modularity_communities(G)
            analysis['communities'] = {
                'num_communities': len(communities),
                'modularity': nx.algorithms.community.modularity(G, communities),
                'largest_community_size': max(len(c) for c in communities) if communities else 0
            }
        except:
            analysis['communities'] = {'error': 'Could not detect communities'}
        
        return analysis
    
    def identify_influential_nodes(self, network_data: NetworkData, top_k: int = 10) -> Dict:
        """
        Identify most influential nodes using various centrality measures
        
        Args:
            network_data: NetworkData object
            top_k: Number of top nodes to return for each measure
            
        Returns:
            Dictionary with top nodes for each centrality measure
        """
        G = nx.from_pandas_edgelist(
            network_data.edges, 
            source='source', 
            target='target', 
            edge_attr='weight',
            create_using=nx.Graph()
        )
        
        if G.number_of_nodes() == 0:
            return {'error': 'Empty graph'}
        
        centrality_measures = {}
        
        # Degree centrality
        degree_cent = nx.degree_centrality(G)
        centrality_measures['degree'] = sorted(
            degree_cent.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        
        # Betweenness centrality
        if G.number_of_nodes() > 2:
            betweenness_cent = nx.betweenness_centrality(G)
            centrality_measures['betweenness'] = sorted(
                betweenness_cent.items(), key=lambda x: x[1], reverse=True
            )[:top_k]
        
        # Closeness centrality
        if nx.is_connected(G):
            closeness_cent = nx.closeness_centrality(G)
            centrality_measures['closeness'] = sorted(
                closeness_cent.items(), key=lambda x: x[1], reverse=True
            )[:top_k]
        
        # PageRank
        pagerank_cent = nx.pagerank(G, weight='weight')
        centrality_measures['pagerank'] = sorted(
            pagerank_cent.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        
        return centrality_measures
    
    def export_analysis_report(self, network_data: NetworkData, output_path: str = None):
        """
        Generate and export a comprehensive analysis report
        
        Args:
            network_data: NetworkData object to analyze
            output_path: Path to save the report (optional)
        """
        # Perform analyses
        structure_analysis = self.analyze_network_structure(network_data)
        influential_nodes = self.identify_influential_nodes(network_data)
        
        # Create report
        report = {
            'network_metadata': network_data.metadata,
            'collection_date': network_data.collection_date.isoformat(),
            'source': network_data.source,
            'structure_analysis': structure_analysis,
            'influential_nodes': influential_nodes,
            'node_statistics': {
                'total_nodes': len(network_data.nodes),
                'features': list(network_data.nodes.columns),
                'label_distribution': network_data.nodes.iloc[:, -1].value_counts().to_dict() if len(network_data.nodes) > 0 else {}
            },
            'edge_statistics': {
                'total_edges': len(network_data.edges),
                'edge_types': network_data.edges.get('relationship_type', pd.Series()).value_counts().to_dict() if len(network_data.edges) > 0 else {},
                'weight_distribution': {
                    'mean': network_data.edges['weight'].mean() if 'weight' in network_data.edges.columns else 0,
                    'std': network_data.edges['weight'].std() if 'weight' in network_data.edges.columns else 0
                }
            }
        }
        
        # Save report if output path specified
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Analysis report saved to {output_path}")
        
        return report

# Example usage and testing
def main():
    """Example usage of the social network data collection framework"""
    
    # Initialize GitHub collector
    # Note: Add your GitHub token for higher rate limits
    github_collector = GitHubCollaborationNetworkCollector(
        github_token=None,  # Add your token here
        rate_limit=0.5  # Be conservative with rate limiting
    )
    
    # Collect network data for a popular repository
    try:
        network_data = github_collector.collect_repository_network(
            repo_owner="tensorflow",
            repo_name="tensorflow",
            max_contributors=50
        )
        
        if network_data:
            # Save the collected data
            network_id = github_collector.save_network_data(network_data)
            print(f"Network data collected and saved with ID: {network_id}")
            
            # Analyze the network
            analyzer = NetworkAnalyzer()
            report = analyzer.export_analysis_report(
                network_data, 
                output_path="tensorflow_network_analysis.json"
            )
            
            print("\n--- Network Analysis Summary ---")
            print(f"Nodes: {report['structure_analysis']['basic_stats']['num_nodes']}")
            print(f"Edges: {report['structure_analysis']['basic_stats']['num_edges']}")
            print(f"Density: {report['structure_analysis']['basic_stats']['density']:.4f}")
            print(f"Average Clustering: {report['structure_analysis']['basic_stats']['average_clustering']:.4f}")
            
            # Print top influential nodes
            if 'pagerank' in report['influential_nodes']:
                print("\n--- Top 5 Most Influential Contributors (PageRank) ---")
                for i, (node, score) in enumerate(report['influential_nodes']['pagerank'][:5], 1):
                    print(f"{i}. {node}: {score:.4f}")
        
        else:
            print("Failed to collect network data")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")

class RedditNetworkCollector(EthicalDataCollector):
    """
    Collect social networks from Reddit using PRAW (Python Reddit API Wrapper)
    """
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str, **kwargs):
        super().__init__(**kwargs)
        
        try:
            import praw
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            logger.info("Reddit API client initialized")
        except ImportError:
            logger.error("PRAW library not installed. Install with: pip install praw")
            raise
    
    def collect_subreddit_network(self, subreddit_name: str, limit: int = 100) -> NetworkData:
        """
        Collect interaction network from a subreddit
        
        Args:
            subreddit_name: Name of subreddit to analyze
            limit: Number of posts to analyze
            
        Returns:
            NetworkData object with comment/post interaction network
        """
        logger.info(f"Collecting network for r/{subreddit_name}")
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            nodes_data = []
            edges_data = []
            user_cache = {}
            
            # Get hot posts from subreddit
            posts = list(subreddit.hot(limit=limit))
            
            for post in posts:
                self.respect_rate_limit()
                
                # Add post author as node
                if post.author and post.author.name not in user_cache:
                    try:
                        author_info = {
                            'user_id': post.author.name,
                            'link_karma': post.author.link_karma,
                            'comment_karma': post.author.comment_karma,
                            'created_utc': post.author.created_utc,
                            'is_premium': getattr(post.author, 'is_gold', False),
                            'verified': getattr(post.author, 'verified', False),
                            'post_karma': post.author.link_karma + post.author.comment_karma
                        }
                        user_cache[post.author.name] = author_info
                        nodes_data.append(author_info)
                    except Exception as e:
                        logger.warning(f"Could not fetch author info for {post.author}: {e}")
                        continue
                
                # Process comments and replies
                post.comments.replace_more(limit=5)  # Limit to avoid too much data
                
                for comment in post.comments.list():
                    if comment.author and comment.author.name != '[deleted]':
                        # Add commenter as node if not already cached
                        if comment.author.name not in user_cache:
                            try:
                                commenter_info = {
                                    'user_id': comment.author.name,
                                    'link_karma': comment.author.link_karma,
                                    'comment_karma': comment.author.comment_karma,
                                    'created_utc': comment.author.created_utc,
                                    'is_premium': getattr(comment.author, 'is_gold', False),
                                    'verified': getattr(comment.author, 'verified', False),
                                    'post_karma': comment.author.link_karma + comment.author.comment_karma
                                }
                                user_cache[comment.author.name] = commenter_info
                                nodes_data.append(commenter_info)
                            except Exception as e:
                                logger.warning(f"Could not fetch commenter info: {e}")
                                continue
                        
                        # Create edge from commenter to post author
                        if post.author and comment.author.name != post.author.name:
                            edges_data.append({
                                'source': comment.author.name,
                                'target': post.author.name,
                                'weight': 1.0,
                                'interaction_type': 'comment_on_post'
                            })
                        
                        # Create edges for comment replies
                        if hasattr(comment, 'parent') and comment.parent().author:
                            parent_author = comment.parent().author.name
                            if parent_author != comment.author.name and parent_author != '[deleted]':
                                edges_data.append({
                                    'source': comment.author.name,
                                    'target': parent_author,
                                    'weight': 0.8,
                                    'interaction_type': 'reply_to_comment'
                                })
            
            # Convert to DataFrames
            nodes_df = pd.DataFrame(nodes_data)
            edges_df = pd.DataFrame(edges_data)
            
            if not nodes_df.empty:
                # Add network features
                nodes_df = self._add_network_features(nodes_df, edges_df)
                
                # Create labels based on karma levels
                nodes_df['activity_level'] = self._categorize_reddit_activity(nodes_df)
            
            metadata = {
                'subreddit': subreddit_name,
                'num_nodes': len(nodes_df),
                'num_edges': len(edges_df),
                'collection_method': 'reddit_praw',
                'features': list(nodes_df.columns) if not nodes_df.empty else [],
                'network_type': 'discussion_forum'
            }
            
            return NetworkData(
                nodes=nodes_df,
                edges=edges_df,
                metadata=metadata,
                collection_date=datetime.now(),
                source='reddit'
            )
            
        except Exception as e:
            logger.error(f"Error collecting Reddit network: {e}")
            return None
    
    def _categorize_reddit_activity(self, nodes_df: pd.DataFrame) -> List[str]:
        """Categorize Reddit users based on karma and activity"""
        if nodes_df.empty:
            return []
        
        labels = []
        for _, node in nodes_df.iterrows():
            total_karma = node.get('post_karma', 0)
            degree = node.get('degree', 0)
            
            # Score based on karma and network position
            activity_score = (
                np.log1p(total_karma) * 0.6 +
                degree * 0.4
            )
            
            if activity_score > 0.8:
                labels.append('highly_active')
            elif activity_score > 0.4:
                labels.append('moderately_active')
            else:
                labels.append('low_activity')
        
        return labels

class LinkedInNetworkCollector(EthicalDataCollector):
    """
    Collect professional networks from LinkedIn
    Note: This is a simplified example - actual LinkedIn API access requires special permissions
    """
    
    def __init__(self, access_token: str, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://api.linkedin.com/v2"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "X-Restli-Protocol-Version": "2.0.0"
        }
        logger.warning("LinkedIn API access is restricted - this is a demonstration")
    
    def collect_company_employee_network(self, company_id: str) -> NetworkData:
        """
        Collect network of employees at a company (if accessible)
        Note: This requires special API access that most developers don't have
        """
        logger.info(f"Attempting to collect network for company {company_id}")
        
        # This is a placeholder implementation
        # Real LinkedIn API access for employee networks is very restricted
        
        nodes_data = [
            {
                'user_id': f'user_{i}',
                'company_id': company_id,
                'position_level': np.random.choice(['entry', 'mid', 'senior', 'executive']),
                'department': np.random.choice(['engineering', 'marketing', 'sales', 'hr']),
                'years_experience': np.random.randint(1, 20),
                'connection_count': np.random.randint(50, 500)
            }
            for i in range(50)  # Simulated data
        ]
        
        # Create some random professional connections
        edges_data = []
        for i in range(len(nodes_data)):
            # Connect to a few random colleagues
            for j in np.random.choice(len(nodes_data), size=np.random.randint(1, 5), replace=False):
                if i != j:
                    edges_data.append({
                        'source': f'user_{i}',
                        'target': f'user_{j}',
                        'weight': 1.0,
                        'relationship_type': 'colleague'
                    })
        
        nodes_df = pd.DataFrame(nodes_data)
        edges_df = pd.DataFrame(edges_data)
        
        # Add network features
        nodes_df = self._add_network_features(nodes_df, edges_df)
        
        # Create seniority-based labels
        nodes_df['seniority_level'] = nodes_df['position_level']
        
        metadata = {
            'company_id': company_id,
            'num_nodes': len(nodes_df),
            'num_edges': len(edges_df),
            'collection_method': 'linkedin_simulation',
            'features': list(nodes_df.columns),
            'network_type': 'professional'
        }
        
        return NetworkData(
            nodes=nodes_df,
            edges=edges_df,
            metadata=metadata,
            collection_date=datetime.now(),
            source='linkedin_demo'
        )

class FacebookNetworkCollector(EthicalDataCollector):
    """
    Facebook/Meta Graph API Collector
    Note: Facebook severely restricts network data access for privacy reasons
    """
    
    def __init__(self, access_token: str, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://graph.facebook.com/v18.0"
        self.headers = {"Authorization": f"Bearer {access_token}"}
        logger.warning("Facebook API access for social graphs is heavily restricted")
    
    def collect_page_engagement_network(self, page_id: str) -> NetworkData:
        """
        Collect engagement network around a Facebook page (if accessible)
        Note: Most Facebook social graph APIs have been deprecated
        """
        logger.info(f"Attempting to collect engagement network for page {page_id}")
        
        # This would require special permissions that are rarely granted
        # Placeholder implementation with simulated data
        
        nodes_data = []
        edges_data = []
        
        # Simulate some page engagement data
        for i in range(100):
            nodes_data.append({
                'user_id': f'user_{i}',
                'engagement_type': np.random.choice(['like', 'comment', 'share']),
                'engagement_frequency': np.random.randint(1, 10),
                'account_age_days': np.random.randint(30, 3650),
                'is_verified': np.random.choice([True, False], p=[0.05, 0.95])
            })
        
        # Create engagement-based connections
        for i in range(len(nodes_data)):
            for j in np.random.choice(len(nodes_data), size=np.random.randint(0, 3), replace=False):
                if i != j:
                    edges_data.append({
                        'source': f'user_{i}',
                        'target': f'user_{j}',
                        'weight': 0.5,
                        'relationship_type': 'co_engagement'
                    })
        
        nodes_df = pd.DataFrame(nodes_data)
        edges_df = pd.DataFrame(edges_data)
        
        nodes_df = self._add_network_features(nodes_df, edges_df)
        nodes_df['engagement_category'] = nodes_df['engagement_type']
        
        metadata = {
            'page_id': page_id,
            'num_nodes': len(nodes_df),
            'num_edges': len(edges_df),
            'collection_method': 'facebook_simulation',
            'features': list(nodes_df.columns),
            'network_type': 'engagement'
        }
        
        return NetworkData(
            nodes=nodes_df,
            edges=edges_df,
            metadata=metadata,
            collection_date=datetime.now(),
            source='facebook_demo'
        )

class NetworkVisualizationTools:
    """
    Tools for visualizing collected social networks
    """
    
    def __init__(self):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            self.plt = plt
            self.sns = sns
            self.sns.set_style("whitegrid")
        except ImportError:
            logger.error("Visualization libraries not installed. Install with: pip install matplotlib seaborn")
    
    def plot_network_graph(self, network_data: NetworkData, output_path: str = None, 
                          layout: str = 'spring', node_size_column: str = None):
        """
        Create a network graph visualization
        
        Args:
            network_data: NetworkData object to visualize
            output_path: Path to save the plot
            layout: Layout algorithm ('spring', 'circular', 'random')
            node_size_column: Column name to use for node sizing
        """
        if not hasattr(self, 'plt'):
            logger.error("Matplotlib not available")
            return
        
        # Create NetworkX graph
        G = nx.from_pandas_edgelist(
            network_data.edges, 
            source='source', 
            target='target', 
            edge_attr='weight',
            create_using=nx.Graph()
        )
        
        # Set up the plot
        fig, ax = self.plt.subplots(1, 1, figsize=(12, 8))
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)
        
        # Node sizes
        if node_size_column and node_size_column in network_data.nodes.columns:
            node_sizes = []
            for node in G.nodes():
                try:
                    size = network_data.nodes[
                        network_data.nodes['user_id'] == node
                    ][node_size_column].iloc[0]
                    node_sizes.append(max(20, min(500, size * 10)))  # Scale and bound
                except:
                    node_sizes.append(50)
        else:
            node_sizes = [50] * len(G.nodes())
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, ax=ax)
        
        # Add labels for high-degree nodes
        high_degree_nodes = [node for node, degree in G.degree() if degree > np.percentile(list(dict(G.degree()).values()), 90)]
        labels = {node: node for node in high_degree_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(f"Network Visualization: {network_data.metadata.get('network_type', 'Unknown')}")
        ax.axis('off')
        
        if output_path:
            self.plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Network visualization saved to {output_path}")
        
        self.plt.show()
    
    def plot_degree_distribution(self, network_data: NetworkData, output_path: str = None):
        """Plot degree distribution of the network"""
        if not hasattr(self, 'plt'):
            logger.error("Matplotlib not available")
            return
        
        G = nx.from_pandas_edgelist(
            network_data.edges, 
            source='source', 
            target='target',
            create_using=nx.Graph()
        )
        
        degrees = [d for n, d in G.degree()]
        
        fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        ax1.hist(degrees, bins=min(50, len(set(degrees))), alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Degree')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Degree Distribution')
        
        # Log-log plot
        degree_counts = pd.Series(degrees).value_counts().sort_index()
        ax2.loglog(degree_counts.index, degree_counts.values, 'bo-', alpha=0.7)
        ax2.set_xlabel('Degree (log scale)')
        ax2.set_ylabel('Frequency (log scale)')
        ax2.set_title('Degree Distribution (Log-Log)')
        
        if output_path:
            self.plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Degree distribution plot saved to {output_path}")
        
        self.plt.show()
    
    def plot_centrality_comparison(self, network_data: NetworkData, output_path: str = None):
        """Compare different centrality measures"""
        if not hasattr(self, 'plt'):
            logger.error("Matplotlib not available")
            return
        
        G = nx.from_pandas_edgelist(
            network_data.edges, 
            source='source', 
            target='target',
            create_using=nx.Graph()
        )
        
        if G.number_of_nodes() < 3:
            logger.warning("Network too small for centrality analysis")
            return
        
        # Calculate centralities
        centralities = {
            'Degree': nx.degree_centrality(G),
            'Betweenness': nx.betweenness_centrality(G),
            'PageRank': nx.pagerank(G)
        }
        
        if nx.is_connected(G):
            centralities['Closeness'] = nx.closeness_centrality(G)
        
        # Create comparison plot
        fig, axes = self.plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, (name, values) in enumerate(centralities.items()):
            if i < len(axes):
                ax = axes[i]
                centrality_values = list(values.values())
                ax.hist(centrality_values, bins=min(20, len(set(centrality_values))), 
                       alpha=0.7, edgecolor='black')
                ax.set_xlabel(f'{name} Centrality')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{name} Centrality Distribution')
        
        # Hide unused subplots
        for i in range(len(centralities), len(axes)):
            axes[i].set_visible(False)
        
        self.plt.tight_layout()
        
        if output_path:
            self.plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Centrality comparison plot saved to {output_path}")
        
        self.plt.show()

# Comprehensive example usage
def comprehensive_example():
    """Comprehensive example showing multiple data sources and analysis"""
    
    print("=== Comprehensive Social Network Analysis Example ===\n")
    
    # Initialize collectors (you would need actual API credentials)
    collectors = {}
    
    try:
        # GitHub collector (works with public repositories)
        collectors['github'] = GitHubCollaborationNetworkCollector(rate_limit=0.5)
        print("✓ GitHub collector initialized")
    except Exception as e:
        print(f"✗ GitHub collector failed: {e}")
    
    # Initialize analyzer and visualizer
    analyzer = NetworkAnalyzer()
    visualizer = NetworkVisualizationTools()
    
    # Collect data from available sources
    networks = {}
    
    if 'github' in collectors:
        try:
            print("\n--- Collecting GitHub Collaboration Network ---")
            networks['github'] = collectors['github'].collect_repository_network(
                repo_owner="matplotlib",
                repo_name="matplotlib", 
                max_contributors=30
            )
            
            if networks['github']:
                print(f"✓ Collected GitHub network: {len(networks['github'].nodes)} nodes, {len(networks['github'].edges)} edges")
                
                # Save data
                network_id = collectors['github'].save_network_data(networks['github'])
                print(f"✓ Data saved with ID: {network_id}")
                
                # Analyze network
                analysis = analyzer.analyze_network_structure(networks['github'])
                influential = analyzer.identify_influential_nodes(networks['github'], top_k=5)
                
                print(f"Network density: {analysis['basic_stats']['density']:.4f}")
                print(f"Average clustering: {analysis['basic_stats']['average_clustering']:.4f}")
                print(f"Connected components: {analysis['connectivity']['num_connected_components']}")
                
                if 'pagerank' in influential:
                    print("\nTop 3 influential contributors:")
                    for i, (user, score) in enumerate(influential['pagerank'][:3], 1):
                        print(f"  {i}. {user}: {score:.4f}")
                
                # Create visualizations (if matplotlib available)
                try:
                    visualizer.plot_degree_distribution(networks['github'])
                    # visualizer.plot_network_graph(networks['github'], layout='spring')
                except Exception as e:
                    print(f"Visualization error: {e}")
                
            else:
                print("✗ Failed to collect GitHub network")
                
        except Exception as e:
            print(f"✗ GitHub collection error: {e}")
    
    # Generate comprehensive report
    if networks:
        print("\n--- Generating Comprehensive Report ---")
        for source, network_data in networks.items():
            if network_data:
                report = analyzer.export_analysis_report(
                    network_data, 
                    output_path=f"{source}_comprehensive_analysis.json"
                )
                print(f"✓ {source.capitalize()} analysis report saved")
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    # Run the main example
    main()
    
    print("\n" + "="*50)
    
    # Run comprehensive example
    comprehensive_example()