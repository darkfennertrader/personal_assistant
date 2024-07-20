#!/bin/bash

# List of packages to install
packages=(
    
    langchain
    langchain-core
    langchain-community
    langchain-experimental
    langchain_openai
    langgraph
    langsmith
    openai
    elevenlabs
    # streamlit
    # streamlit_mic_recorder
    # streamlit-antd-components
)

# Install each package
for package in "${packages[@]}"
do
    echo "Installing $package..."
    pip install -U "$package"
done

echo "All packages installed/updated successfully!"