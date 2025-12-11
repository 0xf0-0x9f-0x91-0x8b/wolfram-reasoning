show_results() {
    for file in $(ls -1tr logs | tail -2); do
        echo "===== FILE: logs/$file ====="
        cat "logs/$file"
        echo
    done
}
