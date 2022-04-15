from annoy import AnnoyIndex





def find_neighbours(query_features , features_list):
    
    f = len(features_list[0])


    t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed

    for i in range(len(features_list)):
        v = features_list[i]
        t.add_item(i, v)

    t.build(50) # 50 trees
    t.save('test.ann')

    # ...

    u = AnnoyIndex(f, 'angular')

    u.load('test.ann') # super fast, will just mmap the file
    indices = u.get_nns_by_vector(query_features,6)
    
    # for ind in indices:
    #     print(features_list[ind])

    return indices
