
        

        
class TestCase(unittest.TestCase):
    
    
    def test_setseed(self):
        for seed in [0.1, -0.4]:
            with duckdb.connect() as conn:
                set_seed(conn, seed)
                result_1 = conn.execute("SELECT random() as random_number FROM range(5);").fetchnumpy()
                result_1 = result_1["random_number"]
                
            with duckdb.connect() as conn:
                set_seed(conn, seed)
                result_2 = conn.execute("SELECT random() as random_number FROM range(5);").fetchnumpy()
                result_2 = result_2["random_number"]
            
            
            for x, y in zip(result_1, result_2):
                assert x == y
        
    
    def test_partition_run(self):
        with dataset_partitioning(number_of_epochs=1, number_of_partions=5) as fetch_data:
            s2, r2 = fetch_data(epoch_idx=0, slice_idx=3)
            
            
        with dataset_partitioning(number_of_epochs=1, number_of_partions=5) as fetch_data:
            s1, r1 = fetch_data(epoch_idx=0, slice_idx=3)

        assert r1[0] == r2[0]
        assert s1[0] == s2[0]
            
            
        assert isinstance(r1[0], str), r1[0]
        assert isinstance(s2[0], int), s2[0]


if __name__ == "__main__":
    unittest.main()
    
    
