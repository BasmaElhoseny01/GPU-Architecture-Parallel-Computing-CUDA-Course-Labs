for (int start_index_x = blockIdx.x * OUTPUT_TILE_DIM - filter_dim / 2; start_index_x < (width + filter_dim / 2); start_index_x += OUTPUT_TILE_DIM)
    {
       for (int start_index_y=blockIdx.y * OUTPUT_TILE_DIM - filter_dim / 2; start_index_y < (height + filter_dim / 2); start_index_y += OUTPUT_TILE_DIM)
        {

     
                if ((start_index_x + threadIdx.x) >= 0 && (start_index_y + threadIdx.y) >= 0 && (start_index_x + threadIdx.x) < width && (start_index_y + threadIdx.y) < height)
                {
                    for (int c = 0; c < IMAGE_CHANNELS; c++)
                    {

                        if ((threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + load_counter * blockDim.x * blockDim.y < INPUT_TILE_DIM * INPUT_TILE_DIM)
                        {
                            sh_mem[(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + load_counter * blockDim.x * blockDim.y] =
                             image[(start_index_y * OUTPUT_TILE_DIM + start_index_x + thread_id) * IMAGE_CHANNELS + c];
         
                        }
                    }
                }
                else
                {
                    // Ghost Padding
                    for (int c = 0; c < IMAGE_CHANNELS; c++)
                    {

                        if ((threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + load_counter * blockDim.x * blockDim.y < INPUT_TILE_DIM * INPUT_TILE_DIM)
                        {
                            // For Threads at end has nothing else to load
                            sh_mem[(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + load_counter * blockDim.x * blockDim.y] = 0.0;
                        }
                    }
                }

                load_counter += 1;
        }
    }
    