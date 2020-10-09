# Main targets:
#   * install: Install Python virtual environment and project dependencies
#   * test: Run all checks on project resources
#   * serve, serve-en: Start local server for English documentation
#   * serve-fr: Start local server for French documentation

check-format: check-format-admonitions check-format-prettier

check-format-admonitions:
	@if grep -IoPrz --exclude-dir='images' \
			'(?<!<!-- prettier-ignore -->)\n(?:\?\?\?|!!!).*?(?:\n|$$)' docs/; then \
		echo && echo 'Above admonitions need <!-- prettier-ignore -->.' 1>&2; \
		exit 1; \
	fi; \
	if grep -IoPrz --exclude-dir='images' \
			'(?<=^|\n)(?:\?\?\?|!!!).*?\n(?!    ).*?(?:\n|$$)' docs/; then \
		echo && echo 'Above admonitions are malformed.' 1>&2; \
		exit 1; \
	fi

check-format-prettier:
	yarn run prettier --ignore-path .gitignore -c .

check-prerequisites:
	@if ! which node npm yarn python3 > /dev/null 2>&1 \
			|| ! python3 -m venv --help > /dev/null 2>&1; then \
		echo -n 'Please ensure the following prerequisites are installed:\n' \
			'  * Node.js (https://github.com/nvm-sh/nvm)\n' \
			'  * npm (comes with Node.js)\n' \
			'  * Yarn (npm install --global yarn)' \
			'  * Python 3.x (apt install python3)\n' \
			'  * Python venv module (apt install python3-venv)\n' 1>&2; \
		exit 1; \
	fi

check-spelling: check-spelling-en check-spelling-fr

check-spelling-en:
	yarn run mdspell --en-gb -trax 'docs/en/*.md' 'docs/en/**/*.md'

check-spelling-fr:
	yarn run mdspell --fr-fr -trax 'docs/fr/*.md' 'docs/fr/**/*.md'

install: install-yarn install-venv

install-venv: check-prerequisites
	python3 -m venv --prompt "$$(pwd | rev | cut -d / -f 1 | rev)" .venv
	. .venv/bin/activate; pip install -Ur requirements.txt

install-yarn: check-prerequisites
	yarn install

serve: serve-en

serve-en:
	. .venv/bin/activate && mkdocs serve -f mkdocs-en.yml

serve-fr:
	. .venv/bin/activate && mkdocs serve -f mkdocs-fr.yml

test: check-format check-spelling

.PHONY: \
	check-format \
	check-format-admonitions \
	check-format-prettier \
	check-prerequisites \
	check-spelling \
	check-spelling-en \
	check-spelling-fr \
	install \
	install-venv \
	install-yarn \
	serve \
	serve-en \
	serve-fr \
	test
