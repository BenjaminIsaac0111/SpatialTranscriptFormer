import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys
import pandas as pd
import tempfile
import shutil

# Add src to python path so we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Correct import based on the project structure
from spatial_transcript_former.data.download import download_metadata, filter_samples, download_hest_subset

class TestDownload(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for testing file operations
        self.test_dir = tempfile.mkdtemp()
        self.metadata_path = os.path.join(self.test_dir, "HEST_v1_3_0.csv")

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    @patch('spatial_transcript_former.data.download.hf_hub_download')
    @patch('os.path.exists')
    def test_download_metadata_exists(self, mock_exists, mock_download):
        # Test case where metadata already exists
        mock_exists.return_value = True
        
        result = download_metadata(self.test_dir)
        
        self.assertEqual(result, self.metadata_path)
        mock_download.assert_not_called()

    @patch('spatial_transcript_former.data.download.hf_hub_download')
    @patch('os.path.exists')
    def test_download_metadata_missing(self, mock_exists, mock_download):
        # Test case where metadata is missing and needs download
        mock_exists.return_value = False
        mock_download.return_value = self.metadata_path
        
        result = download_metadata(self.test_dir)
        
        self.assertEqual(result, self.metadata_path)
        mock_download.assert_called_once()

    def test_filter_samples(self):
        # Create a mock CSV file
        data = {
            'id': ['S1', 'S2', 'S3', 'S4'],
            'organ': ['Bowel', 'Kindey', 'Bowel', 'Lung'],
            'disease_state': ['Cancer', 'Cancer', 'Healthy', 'Cancer'],
            'st_technology': ['Visium', 'Visium', 'Visium', 'Xenium']
        }
        df = pd.DataFrame(data)
        df.to_csv(self.metadata_path, index=False)

        # Test filtering by organ
        samples = filter_samples(self.metadata_path, organ='Bowel')
        self.assertEqual(sorted(samples), ['S1', 'S3'])

        # Test filtering by disease
        samples = filter_samples(self.metadata_path, disease_state='Cancer')
        self.assertEqual(sorted(samples), ['S1', 'S2', 'S4'])

        # Test filtering by multiple criteria
        samples = filter_samples(self.metadata_path, organ='Bowel', disease_state='Cancer')
        self.assertEqual(samples, ['S1'])
        
        # Test no match
        samples = filter_samples(self.metadata_path, organ='Brain')
        self.assertEqual(samples, [])

    @patch('spatial_transcript_former.data.download.snapshot_download')
    def test_download_hest_subset_calls(self, mock_snapshot):
        # Test that snapshot_download is called with correct patterns
        sample_ids = ['S1', 'S2']
        additional_patterns = ['extra_file.txt']
        
        download_hest_subset(sample_ids, self.test_dir, additional_patterns)
        
        mock_snapshot.assert_called_once()
        call_args = mock_snapshot.call_args
        _, kwargs = call_args
        
        self.assertEqual(kwargs['repo_id'], 'MahmoodLab/hest')
        self.assertEqual(kwargs['local_dir'], self.test_dir)
        
        patterns = kwargs['allow_patterns']
        # Check standard recursive patterns
        self.assertIn('**/S1.*', patterns)
        self.assertIn('**/S1_*', patterns)
        self.assertIn('**/S2.*', patterns)
        self.assertIn('README.md', patterns)
        # Check additional patterns
        self.assertIn('extra_file.txt', patterns)

    @patch('spatial_transcript_former.data.download.snapshot_download')
    @patch('zipfile.ZipFile')
    @patch('os.listdir')
    @patch('os.path.exists')
    def test_unzip_logic(self, mock_exists, mock_listdir, mock_zipfile, mock_snapshot):
        # Simulate zip files existing in cellvit_seg
        mock_exists.side_effect = lambda p: p.endswith('cellvit_seg') or p.endswith('xenium_seg') or p.endswith('tissue_seg')
        mock_listdir.return_value = ['file.zip', 'other.txt']
        
        # Mock the zip file context manager
        mock_zip_instance = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
        
        download_hest_subset(['S1'], self.test_dir)
        
        # Check that ZipFile was called for file.zip in checked directories
        # download_hest_subset iterates over ['cellvit_seg', 'xenium_seg', 'tissue_seg']
        # mock_exists returns True for all of them.
        # mock_listdir returns ['file.zip', 'other.txt'] for all of them.
        # So we expect 3 calls (one per directory).
        
        self.assertEqual(mock_zipfile.call_count, 3) 
        mock_zip_instance.extractall.assert_called()

if __name__ == '__main__':
    unittest.main()
