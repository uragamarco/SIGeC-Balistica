#!/usr/bin/env python3
"""
Comprehensive performance tests for the intelligent cache system.
Tests cache performance, memory usage, and different backend configurations.
"""

import os
import sys
import time
import threading
import concurrent.futures
import numpy as np
from pathlib import Path
import tempfile
import shutil
import psutil
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.intelligent_cache import get_cache, initialize_cache
    from image_processing.lbp_cache import get_lbp_cache
    CACHE_AVAILABLE = True
except ImportError as e:
    print(f"Cache system not available: {e}")
    CACHE_AVAILABLE = False

class CachePerformanceTester:
    """Comprehensive cache performance testing suite."""
    
    def __init__(self):
        self.results = {}
        self.temp_dir = None
        self.cache = None
        self.lbp_cache = None
        
    def setup(self):
        """Setup test environment."""
        if not CACHE_AVAILABLE:
            raise RuntimeError("Cache system not available")
            
        # Create temporary directory for cache
        self.temp_dir = tempfile.mkdtemp(prefix="cache_test_")
        
        # Initialize cache with test configuration
        config = {
            'cache_dir': self.temp_dir,
            'max_memory_mb': 100,  # Small for testing
            'ttl_seconds': 300,
            'enable_compression': True,
            'enable_statistics': True
        }
        
        initialize_cache(config)
        self.cache = get_cache()
        self.lbp_cache = get_lbp_cache()
        
        print(f"Cache test setup complete. Temp dir: {self.temp_dir}")
        
    def teardown(self):
        """Cleanup test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temp dir: {self.temp_dir}")
            
    def generate_test_data(self, size_kb=10):
        """Generate test data of specified size."""
        # Generate random data
        data_size = size_kb * 1024
        return np.random.bytes(data_size)
        
    def test_basic_operations(self):
        """Test basic cache operations performance."""
        print("\n=== Testing Basic Cache Operations ===")
        
        results = {
            'set_times': [],
            'get_times': [],
            'hit_rates': [],
            'memory_usage': []
        }
        
        # Test different data sizes
        sizes = [1, 10, 100, 500]  # KB
        num_operations = 100
        
        for size_kb in sizes:
            print(f"Testing {size_kb}KB data...")
            
            # Generate test data
            test_data = self.generate_test_data(size_kb)
            
            # Measure SET operations
            set_times = []
            for i in range(num_operations):
                key = f"test_key_{size_kb}_{i}"
                start_time = time.time()
                self.cache.set(key, test_data, ttl=300)
                set_times.append(time.time() - start_time)
                
            # Measure GET operations
            get_times = []
            hits = 0
            for i in range(num_operations):
                key = f"test_key_{size_kb}_{i}"
                start_time = time.time()
                result = self.cache.get(key)
                get_times.append(time.time() - start_time)
                if result is not None:
                    hits += 1
                    
            hit_rate = hits / num_operations
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            results['set_times'].append({
                'size_kb': size_kb,
                'avg_time': np.mean(set_times),
                'max_time': np.max(set_times),
                'min_time': np.min(set_times)
            })
            
            results['get_times'].append({
                'size_kb': size_kb,
                'avg_time': np.mean(get_times),
                'max_time': np.max(get_times),
                'min_time': np.min(get_times)
            })
            
            results['hit_rates'].append({
                'size_kb': size_kb,
                'hit_rate': hit_rate
            })
            
            results['memory_usage'].append({
                'size_kb': size_kb,
                'memory_mb': memory_usage
            })
            
            print(f"  SET avg: {np.mean(set_times)*1000:.2f}ms")
            print(f"  GET avg: {np.mean(get_times)*1000:.2f}ms")
            print(f"  Hit rate: {hit_rate*100:.1f}%")
            print(f"  Memory: {memory_usage:.1f}MB")
            
        self.results['basic_operations'] = results
        
    def test_concurrent_access(self):
        """Test cache performance under concurrent access."""
        print("\n=== Testing Concurrent Access ===")
        
        num_threads = [1, 2, 4, 8]
        operations_per_thread = 50
        
        results = []
        
        for thread_count in num_threads:
            print(f"Testing with {thread_count} threads...")
            
            # Generate test data
            test_data = self.generate_test_data(10)  # 10KB
            
            def worker_function(thread_id):
                """Worker function for concurrent testing."""
                times = []
                for i in range(operations_per_thread):
                    key = f"concurrent_{thread_id}_{i}"
                    
                    # SET operation
                    start_time = time.time()
                    self.cache.set(key, test_data, ttl=300)
                    times.append(time.time() - start_time)
                    
                    # GET operation
                    start_time = time.time()
                    self.cache.get(key)
                    times.append(time.time() - start_time)
                    
                return times
            
            # Run concurrent test
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(worker_function, i) for i in range(thread_count)]
                all_times = []
                for future in concurrent.futures.as_completed(futures):
                    all_times.extend(future.result())
                    
            total_time = time.time() - start_time
            
            result = {
                'thread_count': thread_count,
                'total_time': total_time,
                'avg_operation_time': np.mean(all_times),
                'operations_per_second': (thread_count * operations_per_thread * 2) / total_time,
                'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024
            }
            
            results.append(result)
            
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Avg operation: {np.mean(all_times)*1000:.2f}ms")
            print(f"  Ops/sec: {result['operations_per_second']:.1f}")
            print(f"  Memory: {result['memory_usage']:.1f}MB")
            
        self.results['concurrent_access'] = results
        
    def test_memory_pressure(self):
        """Test cache behavior under memory pressure."""
        print("\n=== Testing Memory Pressure ===")
        
        # Fill cache beyond configured limit
        large_data = self.generate_test_data(1000)  # 1MB per item
        
        results = {
            'items_stored': 0,
            'evictions_detected': 0,
            'memory_usage': [],
            'performance_degradation': []
        }
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        for i in range(200):  # Try to store 200MB of data
            key = f"memory_test_{i}"
            
            start_time = time.time()
            self.cache.set(key, large_data, ttl=600)
            operation_time = time.time() - start_time
            
            # Check if item was actually stored
            if self.cache.get(key) is not None:
                results['items_stored'] += 1
            else:
                results['evictions_detected'] += 1
                
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            results['memory_usage'].append(current_memory - initial_memory)
            results['performance_degradation'].append(operation_time)
            
            if i % 20 == 0:
                print(f"  Stored {i} items, Memory: {current_memory - initial_memory:.1f}MB")
                
        print(f"  Items successfully stored: {results['items_stored']}")
        print(f"  Evictions detected: {results['evictions_detected']}")
        print(f"  Peak memory usage: {max(results['memory_usage']):.1f}MB")
        print(f"  Avg operation time: {np.mean(results['performance_degradation'])*1000:.2f}ms")
        
        self.results['memory_pressure'] = results
        
    def test_lbp_cache_integration(self):
        """Test LBP cache integration performance."""
        print("\n=== Testing LBP Cache Integration ===")
        
        if not self.lbp_cache:
            print("LBP cache not available")
            return
            
        # Generate test image data (simulate image for LBP processing)
        test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        results = {
            'computation_times': [],
            'cache_hit_times': [],
            'hit_rate': 0
        }
        
        num_operations = 20
        n_points = 8
        radius = 1.0
        
        # Test LBP computation and caching
        computation_times = []
        for i in range(num_operations):
            # Vary the image slightly to test different cache keys
            varied_image = test_image + np.random.randint(-5, 5, test_image.shape, dtype=np.int8)
            varied_image = np.clip(varied_image, 0, 255).astype(np.uint8)
            
            start_time = time.time()
            lbp_result, histogram = self.lbp_cache.get_lbp_pattern(
                varied_image, n_points, radius, compute_histogram=True
            )
            computation_times.append(time.time() - start_time)
            
        # Test cache hits by repeating the same operations
        cache_hit_times = []
        for i in range(num_operations):
            varied_image = test_image + np.random.randint(-5, 5, test_image.shape, dtype=np.int8)
            varied_image = np.clip(varied_image, 0, 255).astype(np.uint8)
            
            start_time = time.time()
            lbp_result, histogram = self.lbp_cache.get_lbp_pattern(
                varied_image, n_points, radius, compute_histogram=True
            )
            cache_hit_times.append(time.time() - start_time)
            
        # Get cache statistics
        cache_stats = self.lbp_cache.get_cache_statistics()
        
        results['computation_times'] = computation_times
        results['cache_hit_times'] = cache_hit_times
        results['hit_rate'] = cache_stats.get('hit_rate', 0)
        
        print(f"  LBP Computation avg: {np.mean(computation_times)*1000:.2f}ms")
        print(f"  LBP Cache hit avg: {np.mean(cache_hit_times)*1000:.2f}ms")
        print(f"  LBP Hit rate: {results['hit_rate']*100:.1f}%")
        print(f"  Cache entries: {cache_stats.get('total_entries', 0)}")
        print(f"  Memory usage: {cache_stats.get('memory_usage_mb', 0):.1f}MB")
        
        self.results['lbp_cache'] = results
        
    def generate_report(self):
        """Generate performance test report."""
        print("\n" + "="*60)
        print("CACHE PERFORMANCE TEST REPORT")
        print("="*60)
        
        # Save results to file
        report_file = os.path.join(self.temp_dir, "cache_performance_report.json")
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"Detailed results saved to: {report_file}")
        
        # Print summary
        if 'basic_operations' in self.results:
            basic = self.results['basic_operations']
            print(f"\nBasic Operations Summary:")
            print(f"  Best SET performance: {min(op['avg_time'] for op in basic['set_times'])*1000:.2f}ms")
            print(f"  Best GET performance: {min(op['avg_time'] for op in basic['get_times'])*1000:.2f}ms")
            print(f"  Overall hit rate: {np.mean([hr['hit_rate'] for hr in basic['hit_rates']])*100:.1f}%")
            
        if 'concurrent_access' in self.results:
            concurrent = self.results['concurrent_access']
            best_throughput = max(result['operations_per_second'] for result in concurrent)
            print(f"\nConcurrency Summary:")
            print(f"  Best throughput: {best_throughput:.1f} ops/sec")
            
        if 'memory_pressure' in self.results:
            memory = self.results['memory_pressure']
            print(f"\nMemory Management Summary:")
            print(f"  Items stored before eviction: {memory['items_stored']}")
            print(f"  Peak memory usage: {max(memory['memory_usage']):.1f}MB")
            
        print(f"\nTest completed successfully!")
        return report_file

def main():
    """Run cache performance tests."""
    if not CACHE_AVAILABLE:
        print("Cache system not available. Please install required dependencies.")
        return 1
        
    tester = CachePerformanceTester()
    
    try:
        tester.setup()
        
        # Run all tests
        tester.test_basic_operations()
        tester.test_concurrent_access()
        tester.test_memory_pressure()
        tester.test_lbp_cache_integration()
        
        # Generate report
        report_file = tester.generate_report()
        
        return 0
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        tester.teardown()

if __name__ == "__main__":
    exit(main())