for filename in phone_models/*/models/*.obj; do
    #echo "$filename"
    #echo "$(basename "$filename" .obj).pcd"
    #echo "$(dirname "${filename}")/$(basename "$filename" .obj)"
    pcl_mesh_sampling "$filename" "$(dirname "${filename}")/$(basename "$filename" .obj).pcd" -n_samples 1000000 -leaf_size 0.001 -write_normals -no_vis_result
done
