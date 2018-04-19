import React from "react";
import Helmet from "react-helmet";
import Link from "gatsby-link";
import styled from "styled-components";

import DefaultNav from "../components/DefaultNav";

import "../styles/normalize.css";
import "../styles/prismjs.css";
import "../styles/base.css";

import favicon from "../images/favicon.ico";

const Container = styled.div`
  max-width: 960px;
  margin: auto;
  padding: 0 10px 0 10px;
`;

class Template extends React.Component {
  render() {
    const { location, children } = this.props;
    let header;
    if (location.pathname === "/") {
      header = <div />;
    } else {
      header = <DefaultNav />;
    }
    return (
      <div>
        <Helmet>
          <link rel="shortcut icon" type="image/png" href={favicon} />
        </Helmet>
        {header}
        <Container>{children()}</Container>
      </div>
    );
  }
}

export default Template;
