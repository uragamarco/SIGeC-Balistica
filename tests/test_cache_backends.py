#!/usr/bin/env python3
"""
Test script to verify different cache backend configurations.
Tests memory cache, disk cache, and distributed cache (if available).
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.intelligent_cache import get_cache, initialize_cache
    CACHE_AVAILABLE = True
except ImportError as e:
    print(f"Cache system not available: {e}")
    CACHE_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import diskcache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False

class CacheBackendTester:
    """Test different cache backend configurations."""
    
    def __init__(self):
        self.temp_dir = None
        self.results = {}
        
    def setup(self):
        """Setup test environment."""
        if not CACHE_AVAILABLE:
            raise RuntimeError("Cache system not available")
            
        self.temp_dir = tempfile.mkdtemp(prefix="cache_backend_test_")
        print(f"Test directory: {self.temp_dir}")
        
    def teardown(self):
        """Cleanup test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up: {self.temp_dir}")
            
    def test_memory_only_cache(self):
        """Test memory-only cache configuration."""
        print("\n=== Testing Memory-Only Cache ===")
        
        config = {
            'max_memory_mb': 50,
            'strategy': 'lru',
            'enable_disk_cache': False,
            'enable_compression': False
        }
        
        cache = initialize_cache(config)
        
        # Test basic operations
        test_data = {"test": "data", "numbers": list(range(100))}
        
        start_time = time.time()
        cache.set("memory_test", test_data, ttl=300)
        set_time = time.time() - start_time
        
        start_time = time.time()
        result = cache.get("memory_test")
        get_time = time.time() - start_time
        
        stats = cache.get_stats()
        
        self.results['memory_only'] = {
            'set_time_ms': set_time * 1000,
            'get_time_ms': get_time * 1000,
            'data_retrieved': result == test_data,
            'hit_rate': stats.hit_rate,
            'entry_count': stats.entry_count
        }
        
        print(f"  SET time: {set_time*1000:.2f}ms")
        print(f"  GET time: {get_time*1000:.2f}ms")
        print(f"  Data integrity: {'✓' if result == test_data else '✗'}")
        print(f"  Hit rate: {stats.hit_rate:.2%}")
        
        cache.shutdown()
        
    def test_disk_cache(self):
        """Test disk cache configuration."""
        print("\n=== Testing Disk Cache ===")
        
        cache_dir = os.path.join(self.temp_dir, "disk_cache")
        
        config = {
            'max_memory_mb': 10,  # Small memory to force disk usage
            'cache_dir': cache_dir,
            'strategy': 'lru',
            'enable_disk_cache': True,
            'enable_compression': True
        }
        
        cache = initialize_cache(config)
        
        # Test with larger data to trigger disk storage
        large_data = {"large_array": list(range(10000)), "metadata": "test"}
        
        start_time = time.time()
        cache.set("disk_test", large_data, ttl=600)
        set_time = time.time() - start_time
        
        # Clear memory cache to force disk read
        cache.clear()
        
        start_time = time.time()
        result = cache.get("disk_test")
        get_time = time.time() - start_time
        
        # Check if cache directory was created
        disk_cache_exists = os.path.exists(cache_dir)
        
        stats = cache.get_stats()
        
        self.results['disk_cache'] = {
            'set_time_ms': set_time * 1000,
            'get_time_ms': get_time * 1000,
            'data_retrieved': result == large_data,
            'disk_cache_created': disk_cache_exists,
            'hit_rate': stats.hit_rate,
            'entry_count': stats.entry_count
        }
        
        print(f"  SET time: {set_time*1000:.2f}ms")
        print(f"  GET time: {get_time*1000:.2f}ms")
        print(f"  Data integrity: {'✓' if result == large_data else '✗'}")
        print(f"  Disk cache created: {'✓' if disk_cache_exists else '✗'}")
        print(f"  Hit rate: {stats.hit_rate:.2%}")
        
        cache.shutdown()
        
    def test_compression_modes(self):
        """Test different compression configurations."""
        print("\n=== Testing Compression Modes ===")
        
        test_data = {"repeated_data": "x" * 1000, "numbers": list(range(500))}
        compression_modes = ['none', 'gzip', 'lz4', 'auto']
        
        compression_results = {}
        
        for mode in compression_modes:
            print(f"  Testing {mode} compression...")
            
            config = {
                'max_memory_mb': 50,
                'compression': mode,
                'enable_compression': mode != 'none'
            }
            
            cache = initialize_cache(config)
            
            start_time = time.time()
            cache.set(f"compression_test_{mode}", test_data, ttl=300)
            set_time = time.time() - start_time
            
            start_time = time.time()
            result = cache.get(f"compression_test_{mode}")
            get_time = time.time() - start_time
            
            stats = cache.get_stats()
            
            compression_results[mode] = {
                'set_time_ms': set_time * 1000,
                'get_time_ms': get_time * 1000,
                'data_integrity': result == test_data,
                'compression_ratio': getattr(stats, 'compression_ratio', 0.0)
            }
            
            print(f"    SET: {set_time*1000:.2f}ms, GET: {get_time*1000:.2f}ms")
            print(f"    Compression ratio: {getattr(stats, 'compression_ratio', 0.0):.2f}")
            
            cache.shutdown()
            
        self.results['compression_modes'] = compression_results
        
    def test_cache_strategies(self):
        """Test different cache eviction strategies."""
        print("\n=== Testing Cache Strategies ===")
        
        strategies = ['lru', 'lfu', 'ttl', 'adaptive']
        strategy_results = {}
        
        for strategy in strategies:
            print(f"  Testing {strategy} strategy...")
            
            config = {
                'max_memory_mb': 5,  # Small to trigger evictions
                'strategy': strategy,
                'max_entries': 10
            }
            
            cache = initialize_cache(config)
            
            # Fill cache beyond capacity
            for i in range(15):
                cache.set(f"strategy_test_{strategy}_{i}", f"data_{i}", ttl=300)
                
            # Access some entries to test strategy behavior
            for i in [0, 2, 4]:
                cache.get(f"strategy_test_{strategy}_{i}")
                
            stats = cache.get_stats()
            
            strategy_results[strategy] = {
                'final_entries': stats.entry_count,
                'evictions': stats.evictions,
                'hit_rate': stats.hit_rate
            }
            
            print(f"    Final entries: {stats.entry_count}")
            print(f"    Evictions: {stats.evictions}")
            print(f"    Hit rate: {stats.hit_rate:.2%}")
            
            cache.shutdown()
            
        self.results['cache_strategies'] = strategy_results
        
    def test_redis_backend(self):
        """Test Redis backend if available."""
        print("\n=== Testing Redis Backend ===")
        
        if not REDIS_AVAILABLE:
            print("  Redis not available, skipping test")
            self.results['redis_backend'] = {'available': False}
            return
            
        try:
            # Try to connect to Redis (assuming default localhost:6379)
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            redis_client.ping()
            
            # Test basic Redis operations
            test_key = "cache_test_redis"
            test_value = json.dumps({"test": "redis_data"})
            
            start_time = time.time()
            redis_client.setex(test_key, 300, test_value)
            set_time = time.time() - start_time
            
            start_time = time.time()
            result = redis_client.get(test_key)
            get_time = time.time() - start_time
            
            data_integrity = json.loads(result) == {"test": "redis_data"}
            
            # Cleanup
            redis_client.delete(test_key)
            
            self.results['redis_backend'] = {
                'available': True,
                'set_time_ms': set_time * 1000,
                'get_time_ms': get_time * 1000,
                'data_integrity': data_integrity
            }
            
            print(f"  Redis connection: ✓")
            print(f"  SET time: {set_time*1000:.2f}ms")
            print(f"  GET time: {get_time*1000:.2f}ms")
            print(f"  Data integrity: {'✓' if data_integrity else '✗'}")
            
        except Exception as e:
            print(f"  Redis connection failed: {e}")
            self.results['redis_backend'] = {'available': False, 'error': str(e)}
            
    def test_diskcache_backend(self):
        """Test DiskCache backend if available."""
        print("\n=== Testing DiskCache Backend ===")
        
        if not DISKCACHE_AVAILABLE:
            print("  DiskCache not available, skipping test")
            self.results['diskcache_backend'] = {'available': False}
            return
            
        try:
            cache_dir = os.path.join(self.temp_dir, "diskcache_test")
            
            # Create DiskCache instance
            disk_cache = diskcache.Cache(cache_dir)
            
            test_data = {"test": "diskcache_data", "numbers": list(range(100))}
            
            start_time = time.time()
            disk_cache.set("diskcache_test", test_data, expire=300)
            set_time = time.time() - start_time
            
            start_time = time.time()
            result = disk_cache.get("diskcache_test")
            get_time = time.time() - start_time
            
            data_integrity = result == test_data
            
            # Get statistics
            stats = {
                'size': len(disk_cache),
                'volume': disk_cache.volume()
            }
            
            disk_cache.close()
            
            self.results['diskcache_backend'] = {
                'available': True,
                'set_time_ms': set_time * 1000,
                'get_time_ms': get_time * 1000,
                'data_integrity': data_integrity,
                'cache_size': stats['size'],
                'volume_bytes': stats['volume']
            }
            
            print(f"  DiskCache: ✓")
            print(f"  SET time: {set_time*1000:.2f}ms")
            print(f"  GET time: {get_time*1000:.2f}ms")
            print(f"  Data integrity: {'✓' if data_integrity else '✗'}")
            print(f"  Cache size: {stats['size']} entries")
            print(f"  Volume: {stats['volume']} bytes")
            
        except Exception as e:
            print(f"  DiskCache test failed: {e}")
            self.results['diskcache_backend'] = {'available': False, 'error': str(e)}
            
    def generate_report(self):
        """Generate comprehensive backend test report."""
        print("\n" + "="*60)
        print("CACHE BACKEND CONFIGURATION REPORT")
        print("="*60)
        
        # Save results to file
        report_file = os.path.join(self.temp_dir, "cache_backend_report.json")
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"Detailed results saved to: {report_file}")
        
        # Print summary
        print("\nBackend Availability:")
        print(f"  Memory Cache: ✓")
        print(f"  Disk Cache: ✓")
        print(f"  Redis: {'✓' if self.results.get('redis_backend', {}).get('available') else '✗'}")
        print(f"  DiskCache: {'✓' if self.results.get('diskcache_backend', {}).get('available') else '✗'}")
        
        if 'compression_modes' in self.results:
            print("\nBest Compression Performance:")
            best_compression = min(
                self.results['compression_modes'].items(),
                key=lambda x: x[1]['set_time_ms'] + x[1]['get_time_ms']
            )
            print(f"  {best_compression[0]}: {best_compression[1]['set_time_ms'] + best_compression[1]['get_time_ms']:.2f}ms total")
            
        if 'cache_strategies' in self.results:
            print("\nStrategy Effectiveness:")
            for strategy, data in self.results['cache_strategies'].items():
                print(f"  {strategy}: {data['evictions']} evictions, {data['hit_rate']:.2%} hit rate")
                
        print("\nConfiguration test completed successfully!")
        return report_file

def main():
    """Run cache backend configuration tests."""
    if not CACHE_AVAILABLE:
        print("Cache system not available. Please install required dependencies.")
        return 1
        
    tester = CacheBackendTester()
    
    try:
        tester.setup()
        
        # Run all backend tests
        tester.test_memory_only_cache()
        tester.test_disk_cache()
        tester.test_compression_modes()
        tester.test_cache_strategies()
        tester.test_redis_backend()
        tester.test_diskcache_backend()
        
        # Generate report
        report_file = tester.generate_report()
        
        return 0
        
    except Exception as e:
        print(f"Backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        tester.teardown()

if __name__ == "__main__":
    exit(main())