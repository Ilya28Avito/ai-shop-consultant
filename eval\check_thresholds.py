
            status = "✅" if value >= threshold else "❌"
            print(f"  {status} {metric}: {value:.2f} (порог: {threshold})")
            if value < threshold:
                failures.append(f"{metric}={value:.2f} < {threshold}")

    return len(failures) == 0, failures


if __name__ == "__main__":
    run = load_latest_run()
    thresholds = load_thresholds()

    print(f"\n📊 Агрегаты прогона:")
    passed, failures = check_thresholds(run, thresholds)

    if passed:
        print("\n✅ Все пороги пройдены — можно деплоить!")
        sys.exit(0)
    else:
        print(f"\n❌ Пороги не пройдены:")
        for f in failures:
            print(f"   - {f}")
        sys.exit(1)
