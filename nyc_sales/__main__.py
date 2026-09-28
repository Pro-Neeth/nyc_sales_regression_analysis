from nyc_sales.pipeline import main

# The guard is required on Windows, where GridSearchCV(n_jobs=-1) starts worker processes
if __name__ == '__main__':
    main()
