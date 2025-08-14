import pytest
import os
from unittest.mock import MagicMock, patch
from project3 import create_dataframe

def test_create_dataframe():
    files = ['smartcity/City1_report.pdf', 'smartcity/City2_report.pdf']
    expected_df = pd.DataFrame({'City': ['City1', 'City2'],
                                'raw_text': ['text from City1_report.pdf', 'text from City2_report.pdf']})
    
    with patch('builtins.open', side_effect=[MagicMock(spec=open, name='file1', read='text from City1_report.pdf'),
                                             MagicMock(spec=open, name='file2', read='text from City2_report.pdf')]):
        with patch('pypdf.PdfReader', side_effect=[MagicMock(spec=pypdf.PdfReader, name='pdf1', pages=[1, 2], _get_page=MagicMock(return_value=MagicMock(spec='Page1'), extract_text=MagicMock(return_value='text from City1_report.pdf'))),
                                                   MagicMock(spec=pypdf.PdfReader, name='pdf2', pages=[1], _get_page=MagicMock(return_value=MagicMock(spec='Page2'), extract_text=MagicMock(return_value='text from City2_report.pdf')))]):
            df = create_dataframe(files)
            pd.testing.assert_frame_equal(df, expected_df)
